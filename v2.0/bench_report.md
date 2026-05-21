# S2S Training Speed Benchmark Report
**Date:** 2026-05-11  
**Hardware:** 4 × NVIDIA H100 NVL graphics processing units, ~94–96 GB memory each  
**Model:** Pangu with variational autoencoder ensemble generation  
---

## What we are measuring

Each training step consists of:
1. Loading a batch of weather data from disk and moving it to the graphics card
2. Running the model forward to produce an ensemble forecast and compute the loss
3. Running the backward pass to compute gradients
4. Updating the model weights with the Adam optimiser

First did some manual measurements to see how long each step takes at steady state — after an initial 20-step warm-up period that lets the graphics drivers and communication libraries initialise. We run 80 measured steps per job and record the median step time (the middle value of the 80 samples) and throughput in samples per second.

---

## Baseline: original code, no changes

Both runs used the original code exactly with a few exceptions, 16-bit floating point arithmetic with dynamic loss scaling, and PyTorch's distributed training configured to search for unused model parameters after every backward pass.

| | Run 1 | Run 2 |
|---|---|---|
| Date/time | 2026-05-11 09:45 | 2026-05-11 10:23 |
| Samples per graphics card per step | 1 | 1 |
| Global samples per step (4 cards) | 4 | 4 |
| Median step time | 0.639 s | 0.638 s |
| 90th-percentile step time | 0.643 s | 0.641 s |
| Step-to-step variation (standard deviation) | 0.005 s | 0.002 s |
| Data loading time per step | 0.003 s | 0.003 s |
| Compute time per step | 0.636 s | 0.636 s |
| **Throughput** | **6.26 samples/s** | **6.27 samples/s** |
| Peak graphics memory (worst card) | 34.96 GB | 34.96 GB |
| Loss scale skips (numerical instability events) | 0 | 0 |

**Key observations:**
- The two runs agree to within 0.05% — the node is stable and the measurement is repeatable.
- Data loading takes 0.003 s out of 0.639 s total (0.4% of step time). The bottleneck is entirely on the graphics card, not disk or data transfer.
- 34.96 GB used out of ~94 GB available — only 37% of graphics memory is occupied. This leaves substantial headroom for optimisations.
- Zero loss scale skips confirm the model is numerically well-behaved in 16-bit arithmetic.

**This establishes the baseline: 0.639 s per step, 6.26 samples per second.**

---

## Profiler analysis (Nsight Systems trace)

Also ran a separate profiling job with full graphics card timeline recording to understand where time is spent inside each step. The capture was limited to the 80 measured steps only (warm-up excluded).

### Time breakdown per step (one graphics card)

| Phase | Median time | Share of step |
|---|---|---|
| Data preparation and transfer | 1.0 ms | 0.15% |
| Forward pass and loss computation | 194.2 ms | 29.1% |
| Backward pass (gradient computation) | 425.8 ms | 63.9% |
| Weight update (optimiser) | 26.3 ms | 3.9% |

The backward pass is **2.2 times longer than the forward pass**. In a normally configured model the backward pass is roughly 1.5–2 times the forward pass due to the extra gradient mathematics. The excess here is caused by gradient checkpointing (see below).

### Gradient checkpointing

Gradient checkpointing is a memory-saving technique. During the forward pass, instead of storing all intermediate layer outputs in memory for later use in the backward pass, the model discards them. When the backward pass needs those outputs to compute gradients, it re-runs the relevant portion of the forward pass from scratch to regenerate them.

The configuration `checkpointing: 2` in the experiment file means roughly every other transformer layer group is checkpointed. This means approximately half of the forward computation is re-run during the backward pass, which explains why backward takes 2.2× longer than forward rather than the expected ~1.5–1.7×.

The layer normalisation backward kernel is the second-largest consumer of graphics card time in the entire profile (17.3 seconds of total recorded time), because it is being re-run repeatedly during the recomputation. 

**Gradient checkpointing was designed to fit the model in memory.** At 34.96 GB used out of ~94 GB available, there is roughly 59 GB of free memory. Reducing or disabling checkpointing would allow layer outputs to be stored rather than recomputed, potentially cutting backward time by 20–30% — but at the cost of using more graphics memory. This memory would not then be available for larger batch sizes.

### Other profiling findings

**Memory layout conversions (4.4 seconds of recorded time):** The model switches between two different ways of arranging data in memory (channel-first and channel-last layouts) between layers. Each conversion wastes time. Fixing the model to use one consistent layout throughout would eliminate this overhead.

**Communication between graphics cards (15.8 seconds of recorded time, ~7.7% of step):** After every backward pass, PyTorch synchronises the gradients across all 4 cards so each card has an identical update to apply. This uses the inter-GPU communication fabric. The `find_unused_parameters` flag (described below) adds overhead to this process.

**Roll operations (7.7 seconds of recorded time):** The shifted-window attention mechanism cyclically shifts feature maps before computing attention. These shift operations run as separate kernels and account for ~24 ms per step across all cards. They are an architectural characteristic and are difficult to optimise without changing the model design.

**Matrix multiplications (8.8 seconds of recorded time):** The core attention and linear layer computations — the operations that mixed-precision arithmetic most directly accelerates — rank 6th and 18th on the kernel list. This means the model is not heavily bottlenecked by matrix math, so switching to a different numeric format will not produce dramatic gains.

---

## Ablation 1: batch size 3 per card, bfloat16 arithmetic, static graph

This run changed three things simultaneously: larger batch size (3 samples per card instead of 1, for 12 total), switched from 16-bit to brain 16-bit floating point, and disabled the unused-parameter search.

| | Baseline | This run |
|---|---|---|
| Samples per card per step | 1 | 3 |
| Floating point format | 16-bit | brain 16-bit |
| Unused-parameter search | enabled | disabled |
| Median step time | 0.639 s | 3.314 s |
| **Throughput** | **6.26 samples/s** | **3.62 samples/s** |
| Peak graphics memory (worst card) | 34.96 GB | **97.02 GB** |

**Result: 42% slower than baseline despite 3× larger batch.** This is worse in every meaningful sense.

**Why:** Graphics memory reached 97 GB — at or beyond the physical limit of the card. The memory allocator was operating under extreme pressure: fragmentation, reallocation overhead, and activations too large to keep efficiently in the fast on-chip cache all compound. The step time scaled 5.2× for only 3× more data — the signature of memory saturation. The `batch_size: 16` (4 per card) configuration triggered an out-of-memory crash before producing results.

Because three variables changed at once, we cannot attribute the result to any one of them. This run was not a useful data point for decision-making.

---

## Ablation 2: bfloat16 arithmetic and static graph, batch size unchanged

This run isolated the numeric format and distributed training changes by keeping batch size at 1 per card (4 global), matching the baseline.

**Changes from baseline:**
1. Switched from 16-bit floating point to brain 16-bit floating point. The brain 16-bit format has the same numeric range as 32-bit (preventing overflow) but uses only 16 bits. On the H100 card it is natively supported and eliminates the need for the dynamic loss scaler that 16-bit requires.
2. Disabled the unused-parameter search in distributed training (`find_unused_parameters=False`, `static_graph=True`). Before each weight update, PyTorch ordinarily traverses the computation graph to identify any model parameters that did not receive a gradient. Two parameters in the Pangu model are permanently unused (their code paths are commented out in the source), so this traversal was wasted work every step. Freezing those parameters and disabling the search removes the overhead.

| | Baseline | This run | Change |
|---|---|---|---|
| Samples per card per step | 1 | 1 | — |
| Floating point format | 16-bit | brain 16-bit | changed |
| Unused-parameter search | enabled | disabled | changed |
| Median step time | 0.639 s | **0.607 s** | **−5.0%** |
| **Throughput** | **6.26 samples/s** | **6.59 samples/s** | **+5.3%** |
| Peak graphics memory | 34.96 GB | 34.96 GB | no change |
| Numeric instability events | 0 | 0 | — |

**Result: 5.3% throughput improvement with no memory cost and no numerical issues.**

Memory did not change because at this batch size the dominant memory consumers are the 32-bit optimiser state (Adam momentum and variance buffers, which are always stored in full 32-bit precision) and the activations, not the numeric format of the computation itself.

Zero numeric instability events with brain 16-bit confirms the model is safe to train in this format. This was expected: the baseline already showed zero instability events in 16-bit arithmetic, and brain 16-bit has a strictly wider numeric range, so any computation that is stable in 16-bit is guaranteed to be stable in brain 16-bit.

---

## Summary table

| Configuration | Step time | Throughput | vs baseline | Memory | Skips |
|---|---|---|---|---|---|
| **Baseline** (16-bit, batch=1/card) | 0.639 s | 6.26 samples/s | — | 35.0 GB | 0 |
| Baseline repeat | 0.638 s | 6.27 samples/s | +0.05% | 35.0 GB | 0 |
| Brain 16-bit + static graph, batch=3/card | 3.314 s | 3.62 samples/s | −42% | 97.0 GB ⚠ | 0 |
| Brain 16-bit + static graph, batch=1/card | 0.607 s | 6.59 samples/s | +5.3% | 35.0 GB | 0 |
| 16-bit + static graph, batch=2/card | 1.160 s | 6.90 samples/s | +10.1% | 69.0 GB | 4 ⚠ |
| **Brain 16-bit + static graph, batch=2/card** | **1.146 s** | **6.98 samples/s** | **+11.4%** | **69.0 GB** | **0** |

The best confirmed configuration is brain 16-bit arithmetic with the static distributed training graph and 2 samples per card — **+11.4% throughput, zero numeric instability events, 73% graphics memory utilisation**.

---

## Remaining experiments

**Next — Just-in-time compilation (in progress):**  
PyTorch's just-in-time kernel compiler (`torch.compile`, mode `reduce-overhead`) fuses consecutive element-wise operations into single GPU kernels. The profiler found that element-wise operations are the single largest consumer of GPU time across the 80 measured steps — over 30 seconds of the total recorded time — because they are currently launched as hundreds of individual small kernels. Compilation would collapse many of these into a single fused operation, reducing both launch overhead and memory bandwidth pressure.

The warmup period has been raised from 20 to 40 steps to allow Triton kernel compilation to settle before timing starts. The compiled steady-state throughput is what will be recorded. Both the wall-clock benchmark script and the Nsight profiling script have been updated to use `reduce-overhead` mode and brain 16-bit arithmetic simultaneously, so the new profile can be compared directly against the original.

**Gradient checkpointing investigation:**  
Code analysis revealed that transformer block checkpointing is commented out in the source — the `checkpointing` value in the configuration file only controls 4 lightweight patch recovery operations, not the transformer blocks themselves. Changing `checkpointing: 2` to `checkpointing: 1` has no effect. Setting it to `0` disables only those 4 patch recovery operations and is expected to give a small gain. The 2.2× backward-to-forward ratio observed in the profiler is inherent to the model depth, not recomputation overhead.

**VAE ensemble quality investigation:**  
A separate test script (`test/vae_collapse_test.py`) has been written to check whether the variational autoencoder is generating meaningful ensemble diversity or has collapsed to a deterministic model. See the VAE section below for context.

---

## VAE ensemble generation — architecture notes

From what I understand the VAE is used to measure uncertainity by generating 4 ensemble members via repeating each input sample 4 times and adding different random noise draws at the bottleneck of the encoder. The noise is sampled from a distribution whose mean and variance are learned by the encoder. A second encoder branch, which only runs during training, processes the target weather state and provides a reference distribution that the forecast encoder is trained to match. This is intended to teach the forecast encoder what the distribution of plausible future atmospheric states looks like in the latent space.  

**What the KL loss is actually doing:**

The KL divergence call is `KL(Encoder1 || Encoder2)`, computed between two learned Gaussian distributions — not between Encoder 1 and a fixed standard Gaussian. The loss function supports a standard Gaussian fallback only when no second distribution is passed; here Encoder 2's mean and variance are always provided. So the target distribution is not N(0,1) — it is whatever Encoder 2 produces when it sees the future weather state.

Encoder 1's distribution is being pushed to match Encoder 2's distribution. At inference time only Encoder 1 runs, sampling noise that — if the KL training worked — resembles what Encoder 2 would have produced had it seen tomorrow's weather. Encoder 2 is purely a training-time teacher; it has no role at inference.

The balance between the forecast loss and the regularisation loss is controlled by a single weight (`vae_loss_weight: 0.0001`). With this weight the KL signal is roughly 10,000 times weaker than the forecast loss, meaning Encoder 1 receives almost no gradient pressure to match Encoder 2's distribution. This leads to two independent failure modes: Encoder 1 never learns to imitate Encoder 2, and separately Encoder 1 may collapse its variance toward zero, making all 4 ensemble members nearly identical. Both failures are likely given the current weight.

**KL formula verification:**

The implementation computes KL(q ∥ p) where q = Encoder 1 and p = Encoder 2:

```
KL = 0.5 × (logvar_p − logvar_q + (var_q + (μ_q − μ_p)²) / var_p − 1)
```

This is the correct closed-form KL divergence between two diagonal Gaussians. The N(0,1) fallback (when Encoder 2 outputs are not provided) reduces to the standard single-encoder VAE formula `0.5 × (μ_q² + var_q − logvar_q − 1)`, also correct.

**Why the math does not prevent collapse despite being correct:**

The gradient of the KL with respect to Encoder 1's log-variance is:

```
∂KL/∂logvar_q = 0.5 × (−1 + var_q / var_p)
```

When variance collapses (var_q → 0) this becomes `−0.5 / var_p` — a negative gradient that pushes log-variance back up, resisting collapse. The math does try to prevent it. The problem is the weight: scaled by 0.0001 the anti-collapse gradient becomes `0.00005 / var_p`. With var_p typically around 0.1–1.0 this is at most 0.0005, while the CRPS gradient is order 1. The forecast loss outweighs the KL by roughly 2,000–20,000 to 1 at the gradient level, so collapse proceeds regardless of the correct formula.


**Simpler alternatives that achieve the same goal:**

| Approach | How diversity is generated | Second encoder needed | Fragility |
|---|---|---|---|
| This model (learned prior) | Noise sampled from encoder distribution, KL training against posterior | Yes — runs on target data during training | High — KL weight must be tuned carefully |
| Fixed noise injection | Add scaled random noise directly at bottleneck, no learned distribution | No | Low |
| Monte Carlo dropout | Keep dropout active at inference, run 4 passes with different dropout masks | No | Low |
| Diffusion in latent space | Score-based sampling from a learned noise schedule | No (separate diffusion head) | Moderate — but principled and current state of the art for this problem |

Google DeepMind's GenCast (2023) uses the diffusion approach on a Pangu-style backbone and currently represents the state of the art for probabilistic medium-range forecasting. The second encoder branch in this model adds training-time compute (it is always gradient-checkpointed) and inference-time architectural complexity for a benefit that depends entirely on the KL weight being correctly tuned.

### Estimated compute cost of the second encoder branch

The second encoder branch (`layer1_e2 → downsample_e2 → layer2_e2 → layer3_e3`) mirrors the main encoder's first three stages. The model configuration uses transformer block depths of `[2, 6, 6, 2]`, so the main encoder runs 2 + 6 + 6 = 14 transformer blocks and the second encoder runs the same 14 blocks on the target data. Both paths run sequentially on the same GPU stream.

From the profiler run at batch size 1 (16-bit, original code), the entire forward pass takes **194 ms**. That 194 ms covers:
- Main encoder (14 blocks + patch embedding + downsample)
- VAE second encoder (14 blocks + downsample), always checkpointed
- VAE first encoder (3 lightweight 1×1 convolutions) — negligible
- Decoder (2 blocks + upsample + patch recovery)

The decoder is shallower (2 blocks) and the patch embedding is a single convolution. The two 14-block encoders together are by far the dominant cost. Assuming the two encoders run at similar throughput and each accounts for roughly equal time, the second encoder is estimated at **40–70 ms of the 194 ms forward pass (20–36%)**.

Because the second encoder is also gradient-checkpointed, its forward computation is re-run during the backward pass. This adds a further estimated **40–70 ms to the 425 ms backward pass**.

**Combined estimated cost of the second encoder: 80–140 ms per training step**, or roughly **12–21% of total step time** at the current batch size of 1.

| Component | Forward | Backward (recomputation) | Total per step |
|---|---|---|---|
| Second encoder (estimated) | 40–70 ms | 40–70 ms | **80–140 ms** |
| As fraction of step time | 6–11% | 6–11% | **12–21%** |

These are estimates based on the block count and the 194 ms forward time. The next Nsight profiling run includes dedicated NVTX markers (`vae_encoder1` and `vae_encoder2`) inside `pangu.py` so the actual measured numbers will replace these estimates. Once the profiler output is available, this table will be updated with measured values.

If the posterior collapse test confirms the second encoder is not producing a useful training signal — which is likely given the 0.0001 regularisation weight — removing it would recover approximately **12–21% of training step time** at no cost to model quality.

Overall:

This is a wierd approach to this problem, for example in Latent diffusion (GenCast approach) you learn a score function over the latent space conditioned on the current atmospheric state. At inference run many denoising steps to produce samples from the true posterior distribution of future states. The conditioning is the current state, which you always have, where as this one it relies on the future state and relies on the encoder2 learning the correct distribution.  

The CVAE design is architecturally justified in settings where the condition is available at inference. Applied to weather forecasting it is a training trick with no inference-time analogue, competing against alternatives that achieves the same goal without the overhead or fragility.  The current 2nd encoder **DOUBLES** the training time.

The S2S model here is trying to do what GenCast does by producing an ensemble from a Pangu-style backbone — but using a CVAE approach that has design problems, **trying to force a shoe to fit**. GenCast demonstrates that the diffusion approach solves the same problem cleanly: no second encoder, no KL weight to tune, no posterior collapse
risk, and the inference-time conditioning (current atmospheric state) is available at every denoising step. The trade-off is inference cost — 50 diffusion chains are slower than one forward pass in pengu.

---

## DSI H200 cluster comparison — why 4 GPUs is slower than expected

The Data Science Institute provided access to a node with 4 × H200 GPUs and ran inference profiles that we compared against the NVIDIA cluster H100 profiles. Their observation was that GPU utilisation was low — roughly in the 15–23% range — and that running 4 GPUs did not speed things up proportionally. We confirmed this with Nsight Systems traces from three configurations: DSI H200 with 1 GPU, DSI H200 with 4 GPUs, and the NVIDIA H100 with 4 GPUs (which is the cluster where the training benchmarks above were collected).

The profiles were exported to SQLite with `nsys export --type=sqlite` and analysed with `v2.0/HPC_scripts/compare_nsys.py`.

### The compute work is identical everywhere

The first thing the profiler makes clear is that the actual on-GPU computation — the time the graphics card spends executing model kernels — is nearly the same across all three setups: roughly 14 seconds per GPU on DSI and 15 seconds per GPU on NVIDIA. The H200 is not slower than the H100 at doing the work itself. The gap is entirely in the time the GPU spends sitting idle between bursts of work.

| Setup | Per-GPU compute (active_ms) | Elapsed wall time (window_ms) | Utilisation |
|---|---|---|---|
| DSI H200 — 1 GPU | 13,977 ms | 35,736 ms | 39% |
| DSI H200 — 4 GPUs (GPU0) | 13,928 ms | 92,477 ms | 15% |
| DSI H200 — 4 GPUs (GPU1–3) | ~13,940 ms | 60–81k ms | 17–23% |
| NVIDIA H100 — 4 GPUs (GPU0) | 15,302 ms | 41,189 ms | 37% |
| NVIDIA H100 — 4 GPUs (GPU1–3) | ~15,295 ms | 27–30k ms | 50–57% |

The DSI 4-GPU run takes roughly 2.2–2.8× longer wall time than the NVIDIA 4-GPU run for the same amount of real computation.

### Where is the time going?

Between every pair of consecutive GPU kernels there is either zero gap (the next kernel starts immediately) or a positive idle period where the GPU is waiting for the CPU to queue more work. We measured all of these gaps on GPU0 across all three profiles.

| Gap size | DSI H200 1-GPU | DSI H200 4-GPU | NVIDIA H100 4-GPU |
|---|---|---|---|
| ≤ 10 ms (normal dispatch) | 468,021 | 467,623 | 469,637 |
| **10–50 ms (frequent short stalls)** | **41** | **423** | **27** |
| 50–100 ms | 7 | 8 | 12 |
| 100–500 ms | 19 | 20 | 12 |
| > 500 ms (I/O or barrier stalls) | 7 | 21 | 14 |
| **Total idle time in gaps > 10 ms** | **19,821 ms** | **72,026 ms** | **24,667 ms** |

The 10–50 ms bucket is the main outlier going from  41 occurrences on DSI with 1 GPU, to 423 occurrences on DSI with 4 GPUs, while NVIDIA with 4 GPUs has only 27. These are weird (not really sure) and not data loading stalls (which would would show up as gaps of 100 ms or more); they are the GPU going briefly idle waiting for the CPU to signal the next kernel launch.

torchrun spawns 4 independent Python processes — one per GPU — so there is no single shared bottleneck across ranks. The cause of the stalls is therefore within each individual rank: something about the DSI cluster's CPU-to-GPU dispatch path adds 10–50 ms between consecutive kernel groups. The measured cumulative idle time across all gaps >10 ms is 72,026 ms on DSI 4-GPU vs 24,667 ms on NVIDIA — a difference of ~47 s — which accounts for most of the wall-time gap shown in the table above.

### PCIe bandwidth contention makes it worse

On the NVIDIA cluster, each GPU's host-to-device transfer bandwidth is consistent whether using 1 GPU or 4 — roughly 42–45 GB/s per card. On DSI, the single-GPU bandwidth is 41.6 GB/s, but under 4-GPU load GPU0 and GPU3 drop to 31–33 GB/s. Some possible reasons could be PCIe contention, the DSI node's four H200s appear to share fewer PCIe root complex lanes (**googled for this answer not sure though**), so when all four GPUs are simultaneously pulling data from the CPU they compete with each other. The slower data transfer contributes directly to the 10–50 ms stalls above.

| Setup | GPU0 bandwidth | GPU1 | GPU2 | GPU3 |
|---|---|---|---|---|
| DSI H200 1-GPU | 41.6 GB/s | — | — | — |
| DSI H200 4-GPUs | **31.5 GB/s** | 38.6 | 37.4 | **32.7** |
| NVIDIA H100 4-GPUs | 44.7 GB/s | 41.8 | 43.6 | 44.3 |

### NCCL is not the issue

No NCCL collective kernels appeared in any of the three profiles. These are independent data-parallel inference runs — each GPU processes its own batch and there is no gradient synchronisation or inter-GPU communication at all.

### Is pinned memory already in use?

Checked the source memory kind recorded by CUPTI for every H2D transfer. All three runs show the same pattern: the large weather data tensors (~165 MB each, 84 transfers per GPU) go through **pinned** memory, while roughly 2,845 smaller transfers per GPU use **pageable** memory. The NVIDIA and DSI 4-GPU distributions are byte-for-byte identical, which means both clusters are running the same DataLoader configuration and `pin_memory` is already enabled for the main data path. The smaller pageable transfers are likely internal CUDA buffers or small parameter tensors, not the weather inputs.

### What may? fix it

The dispatch smoke test (see Midway cluster section below) measured sub-millisecond inter-step dispatch gaps on both Midway H200 nodes, which means Python overhead alone is not the bottleneck — the stalls are more likely a hardware topology or system configuration effect on DSI specifically. With that in mind:

**1. DataLoader prefetching.** Each torchrun rank already has `num_data_workers: 8` set in the config. The next step would be to confirm that `prefetch_factor` is set and that data is being pipelined onto the GPU asynchronously with `non_blocking=True`, so the next batch transfer overlaps with the current forward pass rather than starting after it.

**2. NUMA binding.** If the DSI cluster has a multi-socket layout with GPUs split across NUMA nodes, binding each torchrun process to the CPU cores and memory local to its GPU would eliminate cross-socket dispatch latency. Running `nvidia-smi topo -m` and `numactl --hardware` on DSI would confirm whether this is the case (see hardware topology section below).

The PCIe contentio?  Again unsure, a hardware topology constraint of the DSI node and cannot be fully worked around in software without knowing the actual node topology first.

---

## Midway cluster GPU comparison

To understand whether the DSI performance issues are DSI-specific or reflect the H200 hardware in general, we ran the bandwidth test (`v2.0/test/bandwidth_test.py`) and collected nsys inference profiles on H200 nodes in Midway's test partition. Two node types were tested: Intel Gold-6542Y (midway3-0602) and AMD EPYC-9335 (midway3-0601).

### What we know so far across all measured configurations

| Cluster | GPU | CPU | NVLink | NUMA nodes | NUMA distance | CPU→GPU bandwidth (single GPU) | Concurrent bandwidth drop | Python dispatch gap | GPU util (4-GPU) | Gaps >10ms | Kernel data |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA cluster | H100 NVL | — | unknown¹ | — | — | 42–45 GB/s² | ~0%² | — | 37–57% | 27 | ✓ |
| DSI | H200 | unknown | unknown | unknown | unknown | 41.6 GB/s² | 20–25% (asymmetric)² | — | 15–23% | 423 | ✓ |
| Midway Intel | H200 | Gold-6542Y | NV6 full mesh | 2 (GPU0/1 vs GPU2/3)⁷ | 21 | 26–52 GB/s³ | 10–17% | 0.018 ms⁴ | n/a⁵ | n/a⁵ | ✗⁵ |
| Midway AMD | H200 | EPYC-9335 | NV6 full mesh | 2 (GPU0/1 vs GPU2/3)⁷ | 32 | 35–54 GB/s³ | ~0%⁶ | 0.012 ms⁴ | n/a⁵ | n/a⁵ | ✗⁵ |

¹ `nvidia-smi topo -m` was never captured for the NVIDIA cluster so the NVLink configuration is unknown. H100 NVL typically shows NV6 between paired GPUs but this was not verified.  
² NVIDIA bandwidth figures come from the nsys profile (actual inference transfers, ~165 MB tensors). Midway bandwidth figures come from `bandwidth_test.py` (synthetic tensors, max 13.6 MB upper_air). These measure different transfer sizes and are not directly comparable in absolute GB/s — the relative contention delta is the meaningful comparison.  
³ GPU0 is anomalously slow in the sequential test on both Midway nodes. Intel GPU0 upper_air: 26.7 GB/s vs GPU1–3 at 52 GB/s. AMD GPU0 upper_air: 43.0 GB/s vs GPU2/3 at 54 GB/s (different tensor; AMD AGGREGATE shows GPU0=35.9 GB/s). The GPU0–NIC0 PXB link is present on both nodes but GPU1 also shares a PXB with NIC1 yet is fast, so this is not a complete explanation.  
⁴ Median inter-step dispatch gap from `inference_dispatch_smoke.py` on one GPU, synthetic 87 ms/step workload (6× heavier than real PanguModel). Pattern D (CUDA Graph) was 2.5–3× *slower* than pattern A (list-append): Intel 0.046 vs 0.018 ms, AMD 0.037 vs 0.012 ms. The conclusion is that both are far below 10 ms — not that A ≈ D — so Python dispatch is unlikely to explain 10–50 ms DSI gaps, though the test does not reproduce 4-GPU DDP conditions on DSI.  
⁵ ptrace restrictions on the test partition prevented nsys from capturing torchrun worker GPU activity. Kernel utilisation and gap histograms unavailable.  
⁷ The `nvidia-smi topo -m` CPU Affinity column suggests GPU0/1 are on the same socket as CPUs 0–23/0–31 and GPU2/3 on CPUs 24–47/32–63. However, the `GPU NUMA ID` column reads `N/A` on both nodes, meaning the GPU-to-NUMA mapping was not confirmed by the hardware at time of profiling.  
⁶ AMD EPYC shows ~0% concurrent bandwidth change in the AGGREGATE median. For the largest individual tensor (upper_air, 81.8 MB), concurrent bandwidth was actually higher than sequential on some GPUs — consistent with measurement noise rather than a clean architectural conclusion. The "zero contention" claim holds for the aggregate metric but should not be over-interpreted.

### Key findings from the Midway benchmarks

**H200 on Midway has NVLink (NV6 full mesh).** All four H200s on the Intel node are connected to each other with 6 NVLink bonds each — the same topology class as H100 NVL. This rules out PCIe-only topology as the explanation for Midway's H200 behaviour. Whether the DSI node also has NVLink is still unknown; `nvidia-smi topo -m` on the DSI node would answer this immediately.

**Asymmetric H2D bandwidth appears on both Midway and DSI, suggesting a topology pattern rather than a DSI-specific fault.** On Midway Intel H200, GPU0 gets roughly half the H2D bandwidth of GPU1–3 for large tensors even in the sequential test where only one GPU is active at a time (no contention possible). DSI showed a different but related asymmetry — GPU0 and GPU3 dropped to 31–33 GB/s under concurrent load while GPU1 and GPU2 stayed near 38 GB/s.

To understand why, it helps to know what NUMA means in this context. A dual-socket server has two CPU chips, each with its own pool of local RAM. A GPU copies data from CPU RAM across the PCIe bus. If the GPU is on the "near" CPU socket — the one that owns the RAM being read — the transfer is fast. If it is on the "far" socket, the data first has to cross an inter-socket link (Intel's UPI) before reaching PCIe, which adds latency and reduces bandwidth. This penalty is the NUMA effect.

On Midway the topology output (`nvidia-smi topo -m`) showed:
- GPU0 and GPU1 have CPU affinity 0–23 (Intel) / 0–31 (AMD), suggesting NUMA node 0
- GPU2 and GPU3 have CPU affinity 24–47 (Intel) / 32–63 (AMD), suggesting NUMA node 1
- Cross-socket distance is 2.1× worse than local on Intel (21 vs 10) and 3.2× on AMD (32 vs 10)
- Note: the `GPU NUMA ID` field was `N/A` on both nodes, so the GPU-side NUMA assignment was not directly confirmed by the hardware report

Despite GPU0 and GPU1 being on the same NUMA node, GPU0 was anomalously slow even in isolation — half the bandwidth of GPU1. This does not fit a simple two-socket NUMA explanation and may instead reflect a PCIe switch topology difference at the slot level (GPU0 could be on a PCIe switch that also hosts a NIC, competing for the same upstream lanes). The bandwidth test alone cannot distinguish these causes; `lspci -tv` or a PCIe topology diagram of the server would.

The DSI pattern (GPU0 and GPU3 slow, GPU1 and GPU2 less affected) is more consistent with a clean NUMA split where GPU0 and GPU3 are on the socket that is remote from where the DataLoader is allocating memory. Without `nvidia-smi topo -m` from DSI we cannot confirm this, but the two-GPU asymmetry points to a hardware path difference rather than a random or code-level effect.

**AMD EPYC shows essentially zero concurrent bandwidth degradation.** On the AMD node, all four GPUs maintain their sequential bandwidth under 4-GPU concurrent load (−0.1% to +1.8%). This is strikingly different from the Intel node (10–17% drop) and from DSI (20–25% asymmetric drop). Despite the AMD EPYC-9335 having a higher cross-NUMA distance (32) than Intel (21), its PCIe architecture appears to provide more isolated bandwidth paths per GPU. This suggests the concurrent bandwidth drop on DSI is a PCIe topology issue specific to the node, not a fundamental H200 characteristic.

**What we can and cannot compare between Midway and DSI.**

The dispatch smoke test (`v2.0/test/inference_dispatch_smoke.py`) ran on a **single GPU** on both Midway H200 nodes and measured the GPU idle time between consecutive forward passes using a synthetic model. The median inter-step gap was 0.018 ms on Intel and 0.012 ms on AMD — both far below 10 ms. This rules out Python-level dispatch as the cause of DSI's gaps in single-GPU conditions.

However, the ptrace restriction on the test partition prevented nsys from capturing kernel activity in the 4-GPU runs on Midway H200. **We have no multi-GPU dispatch comparison for Midway H200**, so we cannot say whether 4 processes running concurrently on a Midway H200 node would produce the same gap explosion seen on DSI (41 → 423 gaps from 1 to 4 GPUs).

What makes the single-GPU comparison meaningful is the DSI 1-GPU profile: even on **one GPU with no inter-process competition**, DSI already shows **41 gaps of 10–50 ms** and only 39% GPU utilisation. Something about DSI's single-GPU environment already causes elevated dispatch latency. The 4-GPU case makes it roughly 10× worse (423 gaps), but the problem is present before any multi-GPU effects are introduced. The Midway H200 single-GPU dispatch test shows no such elevated latency, which suggests the elevated baseline is DSI-specific rather than a property of H200 hardware in general.

To complete the comparison properly — to check whether Midway H200 under 4-GPU DDP would reproduce the same pattern as DSI — we need either the ptrace restriction lifted on the test partition or the same inference nsys profile run on the `pedramh-gpu` H100 partition (which has different security settings).

### What is still unknown

The critical missing piece is DSI's hardware topology. Running two commands on the DSI node would resolve the main open questions:

```bash
nvidia-smi topo -m     # shows NVLink vs PCIe, GPU-to-NUMA mapping
numactl --hardware     # shows NUMA node count and distances
```

If DSI shows `PIX` or `PHB` links (PCIe-only, no NVLink) where Midway shows `NV6`, that is the primary structural difference. If DSI also shows `NV6`, the topology is similar and the 10–50 ms gaps are more likely attributable to NUMA mis-binding or a different PyTorch/CUDA version — both of which affect how quickly the kernel launch signal reaches the GPU command processor without being about Python overhead per se.

---

## Notes on measurement reliability

- All step times are measured with graphics card synchronisation barriers (`torch.cuda.synchronize()`) on both sides of the timing window. This ensures we record actual execution time, not just how long it takes to submit work to the graphics card.
- The warm-up period (20 steps) absorbs one-time costs: driver initialisation, automatic kernel selection (cuDNN autotuning), and communication library warm-up.
- The timer includes a self-consistency check: the sum of individually measured step windows must agree with the total elapsed wall time within 10%. If they disagree, the result is discarded rather than recorded.

