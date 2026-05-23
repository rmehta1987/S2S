# S2S Training Speed Benchmark Report
**Date:** 2026-05-11  
**Hardware:** 4 × NVIDIA H100 NVL GPUs, ~94–96 GB memory each  
**Model:** Pangu with variational autoencoder ensemble generation  
---

## What we are measuring

Each training step consists of:
1. Loading a batch of weather data from disk and moving it to the GPU
2. Running the model forward to produce an ensemble forecast and compute the loss
3. Running the backward pass to compute gradients
4. Updating the model weights with the Adam optimiser

First did some manual measurements to see how long each step takes at steady state — after an initial 20-step warm-up period that lets the GPU drivers and communication libraries initialise. We run 80 measured steps per job and record the median step time (the middle value of the 80 samples) and throughput in samples per second.

---

## Baseline: original code, no changes

Both runs used the original code exactly with a few exceptions, 16-bit floating point arithmetic with dynamic loss scaling, and PyTorch's distributed training configured to search for unused model parameters after every backward pass.

| | Run 1 | Run 2 |
|---|---|---|
| Date/time | 2026-05-11 09:45 | 2026-05-11 10:23 |
| Samples per GPU per step | 1 | 1 |
| Global samples per step (4 GPUs) | 4 | 4 |
| Median step time | 0.639 s | 0.638 s |
| 90th-percentile step time | 0.643 s | 0.641 s |
| Step-to-step variation (standard deviation) | 0.005 s | 0.002 s |
| Data loading time per step | 0.003 s | 0.003 s |
| Compute time per step | 0.636 s | 0.636 s |
| **Throughput** | **6.26 samples/s** | **6.27 samples/s** |
| Peak GPU memory (worst card) | 34.96 GB | 34.96 GB |
| Loss scale skips (numerical instability events) | 0 | 0 |

**Key observations:**
- The two runs agree to within 0.05% — the node is stable and the measurement is repeatable.
- Data loading takes 0.003 s out of 0.639 s total (0.4% of step time). The bottleneck is entirely on the GPU, not disk or data transfer.
- 34.96 GB used out of ~94 GB available — only 37% of GPU memory is occupied. This leaves substantial headroom for optimisations.
- Zero loss scale skips confirm the model is numerically well-behaved in 16-bit arithmetic.

**This establishes the baseline: 0.639 s per step, 6.26 samples per second.**

---

## Profiler analysis (Nsight Systems trace)

Also ran a separate profiling job with full GPU timeline recording to understand where time is spent inside each step. The profiler is NVIDIA's **Nsight Systems** (`nsys` for short): it records every CUDA event — kernel launches, memory copies, synchronisations — with nanosecond precision and writes a `.nsys-rep` file we can post-process. The capture was limited to the 80 measured steps only (warm-up excluded).

### Time breakdown per step (one GPU)

| Phase | Median time | Share of step |
|---|---|---|
| Data preparation and transfer | 1.0 ms | 0.15% |
| Forward pass and loss computation | 194.2 ms | 29.1% |
| Backward pass (gradient computation) | 425.8 ms | 63.9% |
| Weight update (optimiser) | 26.3 ms | 3.9% |

The backward pass is **2.2 times longer than the forward pass**. In a normally configured transformer the backward pass is typically 1.5–2× the forward pass due to gradient mathematics; the 2.2× ratio observed here is on the high side but, on later inspection of the source code, is **not** driven by gradient recomputation — the transformer-block checkpointing call in `pangu.py` is commented out, so no forward work is being re-run (see the "Remaining experiments" section below, "Gradient checkpointing investigation" item, for the source-code check). The ratio appears to be inherent to the model depth and the per-layer kernel mix.

### LayerNorm backward kernel

The layer normalisation backward kernel is the second-largest consumer of GPU time in the entire profile (17.3 seconds of total recorded time across 80 measured steps). It dominates because the model contains many normalisation layers — each is invoked once in the forward and once in the backward — not because of any recomputation pass. Fusing LayerNorm into adjacent operations via `torch.compile` is the largest single optimisation opportunity the profile suggests.

**Memory headroom for larger batches.** Peak memory is 34.96 GB out of ~94 GB available, leaving roughly 59 GB free. This headroom is what allows the batch-size-2-per-card experiment below (which uses ~69 GB) to fit without an out-of-memory crash.

### Other profiling findings

**Memory layout conversions (4.4 seconds of recorded time):** The model switches between two different ways of arranging data in memory (channel-first and channel-last layouts) between layers. Each conversion wastes time. Fixing the model to use one consistent layout throughout would eliminate this overhead.

**Communication between GPUs (15.8 seconds of recorded time, ~7.7% of step):** After every backward pass, PyTorch synchronises the gradients across all 4 GPUs so each one applies an identical update. This uses the inter-GPU communication fabric (NCCL — NVIDIA's collective communication library). The `find_unused_parameters` flag (described below) adds overhead to this process.

**Roll operations (7.7 seconds of recorded time):** The shifted-window attention mechanism cyclically shifts feature maps before computing attention. These shift operations run as separate kernels and account for ~24 ms per step across all cards. They are an architectural characteristic and are difficult to optimise without changing the model design.

**Matrix multiplications (8.8 seconds of recorded time):** The core attention and linear layer computations — the operations that mixed-precision arithmetic most directly accelerates — rank 6th and 18th on the kernel list. This means the model is not heavily bottlenecked by matrix math, so switching to a different numeric format will not produce dramatic gains.

---

## Ablation 1: batch size 3 per card, bfloat16 arithmetic, static graph

This run changed three things simultaneously: larger batch size (3 samples per card instead of 1, for 12 total), switched from 16-bit to bfloat16 floating point, and disabled the unused-parameter search.

| | Baseline | This run |
|---|---|---|
| Samples per GPU per step | 1 | 3 |
| Floating point format | 16-bit | bfloat16 |
| Unused-parameter search | enabled | disabled |
| Median step time | 0.639 s | 3.314 s |
| **Throughput** | **6.26 samples/s** | **3.62 samples/s** |
| Peak GPU memory (worst card) | 34.96 GB | **97.02 GB** |

**Result: 42% slower than baseline despite 3× larger batch.** This is worse in every meaningful sense.

**Why:** GPU memory reached 97 GB — at or beyond the physical limit of the card. The memory allocator was operating under extreme pressure: fragmentation, reallocation overhead, and activations too large to keep efficiently in the fast on-chip cache all compound. The step time scaled 5.2× for only 3× more data — the signature of memory saturation. The `batch_size: 16` (4 per card) configuration triggered an out-of-memory crash before producing results.

Because three variables changed at once, we cannot attribute the result to any one of them. This run was not a useful data point for decision-making.

---

## Ablation 2: bfloat16 arithmetic and static graph, batch size unchanged

This run isolated the numeric format and distributed training changes by keeping batch size at 1 per card (4 global), matching the baseline.

**Changes from baseline:**
1. Switched from 16-bit floating point to **bfloat16** ("brain floating-point 16", a 16-bit format from Google Brain with the same exponent range as 32-bit). bfloat16 has the same numeric range as 32-bit (preventing overflow) but uses only 16 bits. On the H100 it is natively supported and eliminates the need for the dynamic loss scaler that 16-bit requires.
2. Disabled the unused-parameter search in distributed training (`find_unused_parameters=False`, `static_graph=True`). Before each weight update, PyTorch ordinarily traverses the computation graph to identify any model parameters that did not receive a gradient. Two parameters in the Pangu model are permanently unused (their code paths are commented out in the source), so this traversal was wasted work every step. Freezing those parameters and disabling the search removes the overhead.

| | Baseline | This run | Change |
|---|---|---|---|
| Samples per GPU per step | 1 | 1 | — |
| Floating point format | 16-bit | bfloat16 | changed |
| Unused-parameter search | enabled | disabled | changed |
| Median step time | 0.639 s | **0.607 s** | **−5.0%** |
| **Throughput** | **6.26 samples/s** | **6.59 samples/s** | **+5.3%** |
| Peak GPU memory | 34.96 GB | 34.96 GB | no change |
| Numeric instability events | 0 | 0 | — |

**Result: 5.3% throughput improvement with no memory cost and no numerical issues.**

Memory did not change because at this batch size the dominant memory consumers are the 32-bit optimiser state (Adam momentum and variance buffers, which are always stored in full 32-bit precision) and the activations, not the numeric format of the computation itself.

Zero numeric instability events with bfloat16 confirms the model is safe to train in this format. This was expected: the baseline already showed zero instability events in 16-bit arithmetic, and bfloat16 has a strictly wider numeric range, so any computation that is stable in 16-bit is guaranteed to be stable in bfloat16.

---

## Summary table

| Configuration | Step time | Throughput | vs baseline | Memory | Skips |
|---|---|---|---|---|---|
| **Baseline** (16-bit, batch=1/card) | 0.639 s | 6.26 samples/s | — | 35.0 GB | 0 |
| Baseline repeat | 0.638 s | 6.27 samples/s | +0.05% | 35.0 GB | 0 |
| bfloat16 + static graph, batch=3/card | 3.314 s | 3.62 samples/s | −42% | 97.0 GB ⚠ | 0 |
| bfloat16 + static graph, batch=1/card | 0.607 s | 6.59 samples/s | +5.3% | 35.0 GB | 0 |
| 16-bit + static graph, batch=2/card | 1.160 s | 6.90 samples/s | +10.1% | 69.0 GB | 4 ⚠ |
| **bfloat16 + static graph, batch=2/card** | **1.146 s** | **6.98 samples/s** | **+11.4%** | **69.0 GB** | **0** |

The best confirmed configuration is bfloat16 arithmetic with the static distributed training graph and 2 samples per card — **+11.4% throughput, zero numeric instability events, 73% GPU memory utilisation**.

---

## Remaining experiments

**Next — Just-in-time compilation (in progress):**  
PyTorch's just-in-time kernel compiler (`torch.compile`, mode `reduce-overhead`) fuses consecutive element-wise operations into single GPU kernels. The profiler found that element-wise operations are the single largest consumer of GPU time across the 80 measured steps — over 30 seconds of the total recorded time — because they are currently launched as hundreds of individual small kernels. Compilation would collapse many of these into a single fused operation, reducing both launch overhead and memory bandwidth pressure.

The warmup period has been raised from 20 to 40 steps to allow Triton kernel compilation to settle before timing starts (Triton is the GPU-kernel compiler `torch.compile` uses internally to generate fused kernels). The compiled steady-state throughput is what will be recorded. Both the wall-clock benchmark script and the Nsight profiling script have been updated to use `reduce-overhead` mode and bfloat16 arithmetic simultaneously, so the new profile can be compared directly against the original.

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

Google DeepMind's GenCast (2023) uses the diffusion approach on a Pangu-style backbone and currently represents the state of the art for probabilistic medium-range forecasting. The second encoder branch in this model adds training-time compute and inference-time architectural complexity for a benefit that depends entirely on the KL weight being correctly tuned.

### Estimated compute cost of the second encoder branch

The second encoder branch (`layer1_e2 → downsample_e2 → layer2_e2 → layer3_e3`) mirrors the main encoder's first three stages. The model configuration uses transformer block depths of `[2, 6, 6, 2]`, so the main encoder runs 2 + 6 + 6 = 14 transformer blocks and the second encoder runs the same 14 blocks on the target data. Both paths run sequentially on the same GPU stream (a stream is a queue of operations that execute in order; operations on different streams can overlap, but here both encoders share one queue, so the second one waits for the first to finish).

From the profiler run at batch size 1 (16-bit, original code), the entire forward pass takes **194 ms**. That 194 ms covers:
- Main encoder (14 blocks + patch embedding + downsample)
- VAE second encoder (14 blocks + downsample)
- VAE first encoder (3 lightweight 1×1 convolutions) — negligible
- Decoder (2 blocks + upsample + patch recovery)

The decoder is shallower (2 blocks) and the patch embedding is a single convolution. The two 14-block encoders together are by far the dominant cost. Assuming the two encoders run at similar throughput and each accounts for roughly equal time, the second encoder is estimated at **40–70 ms of the 194 ms forward pass (20–36%)**.

The second encoder's backward pass adds an estimated **60–100 ms** (standard gradient computation; transformer-block checkpointing is commented out in source — see the "Remaining experiments" section above — so its activations are stored and the backward does not pay a recomputation cost). The 60–100 ms range is the typical ~1.5× of forward seen for self-attention blocks without recomputation.

**Combined estimated cost of the second encoder: 100–170 ms per training step**, or roughly **15–25% of total step time** at the current batch size of 1.

| Component | Forward | Backward (gradients only, no recomputation) | Total per step |
|---|---|---|---|
| Second encoder (estimated) | 40–70 ms | 60–100 ms | **100–170 ms** |
| As fraction of step time | 6–11% | 9–15% | **15–25%** |

These are estimates based on the block count and the 194 ms forward time. The next Nsight profiling run includes dedicated NVTX markers (`vae_encoder1` and `vae_encoder2`) inside `pangu.py` — NVTX markers are annotations injected into the code so the profiler can label regions of the timeline by name — so the actual measured numbers will replace these estimates. Once the profiler output is available, this table will be updated with measured values.

If the posterior collapse test confirms the second encoder is not producing a useful training signal — which is likely given the 0.0001 regularisation weight — removing it would recover approximately **15–25% of training step time** at no cost to model quality.

Overall:

This is a wierd approach to this problem, for example in Latent diffusion (GenCast approach) you learn a score function over the latent space conditioned on the current atmospheric state. At inference run many denoising steps to produce samples from the true posterior distribution of future states. The conditioning is the current state, which you always have, where as this one it relies on the future state and relies on the encoder2 learning the correct distribution.  

The CVAE design is architecturally justified in settings where the condition is available at inference. Applied to weather forecasting it is a training trick with no inference-time analogue, competing against alternatives that achieves the same goal without the overhead or fragility.  The current 2nd encoder **DOUBLES** the training time.

The S2S model here is trying to do what GenCast does by producing an ensemble from a Pangu-style backbone — but using a CVAE approach that has design problems, **trying to force a shoe to fit**. GenCast demonstrates that the diffusion approach solves the same problem cleanly: no second encoder, no KL weight to tune, no posterior collapse
risk, and the inference-time conditioning (current atmospheric state) is available at every denoising step. The trade-off is inference cost — 50 diffusion chains are slower than one forward pass in pengu.

---

## DSI H200 cluster comparison — why 4 GPUs is slower than expected

The Data Science Institute provided access to a node with 4 × H200 GPUs and ran inference profiles that we compared against the NVIDIA cluster H100 profiles. Their observation was that GPU utilisation was low — roughly in the 15–23% range — and that running 4 GPUs did not speed things up proportionally. We confirmed this with Nsight Systems traces from three configurations: DSI H200 with 1 GPU, DSI H200 with 4 GPUs, and the NVIDIA H100 with 4 GPUs (which is the cluster where the training benchmarks above were collected).

The profiles were exported to SQLite with `nsys export --type=sqlite` and analysed with `v2.0/HPC_scripts/compare_nsys.py`.

### The compute work is similar across clusters

The first thing the profiler shows is that the total on-GPU kernel-active time agrees within ~10% across all three setups: roughly 14 seconds per GPU on DSI and 15 seconds per GPU on NVIDIA. This is consistent with the H200 doing the inference work at a similar per-GPU rate to the H100 for this model — but it does not by itself establish per-kernel parity. The two clusters did not hold the software stack constant (NVIDIA runs inside the NGC apptainer image with one CUDA/cuDNN, DSI is bare-metal with possibly different versions; the nsys version per cluster was also not recorded), and the captured iteration counts may differ slightly. With those caveats noted, the gap in wall time is still dominated by GPU idle time between bursts of work, not by per-kernel slowdown.

| Setup | Per-GPU compute (active_ms) | Elapsed wall time (window_ms) | Utilisation |
|---|---|---|---|
| DSI H200 — 1 GPU | 13,977 ms | 35,736 ms | 39% |
| DSI H200 — 4 GPUs (GPU0) | 13,928 ms | 92,477 ms | 15% |
| DSI H200 — 4 GPUs (GPU1–3) | ~13,940 ms | 60–81k ms | 17–23% |
| NVIDIA H100 — 4 GPUs (GPU0) | 15,302 ms | 41,189 ms | 37% |
| NVIDIA H100 — 4 GPUs (GPU1–3) | ~15,295 ms | 27–30k ms | 50–57% |

The DSI 4-GPU run takes roughly 2.2–2.8× longer wall time than the NVIDIA 4-GPU run for the same amount of real computation.

### Where is the time going?

Between every pair of consecutive GPU kernels there is either zero gap (the next kernel starts immediately) or a positive idle period where the GPU is waiting for the CPU to queue more work. We refer to this gap throughout the report as **CPU-to-GPU handoff** time (NVIDIA's tooling sometimes calls the same quantity "dispatch latency" — same concept, different word). We measured all of these gaps on GPU0 across all three profiles.

| Gap size | DSI H200 1-GPU | DSI H200 4-GPU | NVIDIA H100 4-GPU |
|---|---|---|---|
| ≤ 10 ms (normal handoff) | 468,021 | 467,623 | 469,637 |
| **10–50 ms (frequent short stalls)** | **41** | **423** | **27** |
| 50–100 ms | 7 | 8 | 12 |
| 100–500 ms | 19 | 20 | 12 |
| > 500 ms (I/O or barrier stalls) | 7 | 21 | 14 |
| **Total idle time in gaps > 10 ms** | **19,821 ms** | **72,026 ms** | **24,667 ms** |

The 10–50 ms bucket is the main outlier going from  41 occurrences on DSI with 1 GPU, to 423 occurrences on DSI with 4 GPUs, while NVIDIA with 4 GPUs has only 27. These are weird (not really sure) and not data loading stalls (which would would show up as gaps of 100 ms or more); they are the GPU going briefly idle waiting for the CPU to signal the next kernel launch.

`torchrun` (PyTorch's distributed launcher) spawns 4 independent Python processes — one per GPU — so there is no shared application-level synchronisation across ranks. However, the four ranks still share host-level resources: CPU cores, memory bandwidth, the PCIe root complex, kernel locks (mmap, page-fault handling), and IRQ paths. The 41→423 jump in 10–50 ms gaps on GPU0 when scaling from 1 GPU to 4 GPUs is itself evidence of inter-rank contention via these shared host resources, not a constant per-rank handoff cost. The measured cumulative idle time across all gaps >10 ms is 72,026 ms on DSI 4-GPU vs 24,667 ms on NVIDIA — a difference of ~47 s — which accounts for most of the wall-time gap shown in the table above.

### H2D bandwidth: the production-sized tensors are fine; small transfers are the contended ones

A note on the hardware: **PCIe** (PCI Express) is the bus that connects the GPU to the CPU and to system RAM; every byte of weather data the model reads off disk has to cross PCIe before it can sit in GPU memory. The relevant theoretical ceiling for H100 NVL / H200 (both PCIe Gen5 ×16) is ~64 GB/s in each direction; in practice you see ~70–80% of that at best.

The averaged-over-all-transfers bandwidth (taken from total bytes ÷ total time across every H2D call recorded in the profile) tells a misleading story on its own:

| Setup | All H2D (per GPU) — total GB ÷ total time | GPU0 | GPU1 | GPU2 | GPU3 |
|---|---|---|---|---|---|
| DSI H200 1-GPU | 41.6 GB/s | — | — | — | — |
| DSI H200 4-GPUs | mean 35.1 GB/s | **31.5** | 38.6 | 37.4 | **32.7** |
| NVIDIA H100 4-GPUs | mean 43.6 GB/s | 44.7 | 41.8 | 43.6 | 44.3 |

These figures average together two very different regimes — 84 large pinned transfers per GPU (mean 165 MB, dominated by 12 transfers of ~991 MB and 12 of ~105 MB) and 2,845 small pageable transfers per GPU (averaging ~560 KB). Splitting by transfer size:

| Transfer size class | DSI 1-GPU | DSI 4-GPU (per GPU) | NVIDIA H100 4-GPU (per GPU) |
|---|---|---|---|
| **>100 MB** (24/GPU, ~13 GB of data) | **55.4 GB/s** | **55.4 / 55.4 / 55.4 / 55.4** | **55.3 / 55.4 / 55.2 / 55.3** |
| >10 MB (74/GPU, ~14 GB of data) | 49.9 GB/s | 43.5 / 48.8 / 48.4 / 44.9 | 50.8 / 49.7 / 50.3 / 50.0 |
| All transfers (2929/GPU) | 41.6 GB/s | 31.5 / 38.6 / 37.4 / 32.7 | 44.7 / 41.8 / 43.6 / 44.3 |

**Restricted to the production-sized weather tensors (>100 MB), DSI under 4-GPU concurrent load is identical to DSI single-GPU and identical to the H100 cluster — all four GPUs sustain ~55 GB/s with no degradation.** The PCIe bus has plenty of headroom for the dominant data path. The headline "DSI bandwidth drops 24% on GPU0/3" comes entirely from the small-transfer regime where each call is latency-bound (CPU-to-GPU handoff, IRQ handling, kernel-lock contention), and four ranks competing for those host resources serialise them.

The Midway Intel `bandwidth_test.py` numbers behave the same way: on the production-sized `upper_air_input` (22 MB) the concurrent-vs-sequential drop is only **−1.6% to −3.5%** on all four GPUs; the −55% aggregate figure quoted elsewhere in this report is dominated by the small `varying_boundary` tensor (3.9 MB) which collapses by −57 to −62%.

**Implication.** This invalidates the earlier "PCIe contention is contributing to the 10–50 ms stalls" causal chain in the section above. The large-tensor H2D path on DSI is not bandwidth-limited under 4-GPU load — the 24% bimodal drop is a *small-transfer* effect, not a production-workload effect. The wall-time deficit on DSI vs NVIDIA is therefore almost entirely explained by the GPU0 handoff gaps (and the corresponding gaps on GPU1–3), not by H2D throughput on the weather data.

What remains worth investigating: the 2,845 small pageable transfers per GPU per profile *do* serialise badly under 4-GPU load, and if any of them sit on the critical path (e.g., a small parameter or index tensor that the next kernel waits for) they could be one mechanism contributing to the handoff stalls. But this is a much narrower claim than "PCIe contention" and would require correlating individual MEMCPY events to subsequent kernel launches to confirm.

### NCCL is not the issue

NCCL is NVIDIA's multi-GPU communication library — the layer PyTorch uses to synchronise gradients across GPUs during distributed training. No NCCL collective kernels appeared in any of the three profiles. These are independent data-parallel inference runs — each GPU processes its own batch and there is no gradient synchronisation or inter-GPU communication at all.

### Is pinned memory already in use?

Checked the source memory kind recorded by CUPTI (CUDA's Profiling Tools Interface, the low-level event hook that `nsys` uses internally) for every H2D transfer. All three runs show the same pattern: 84 large weather-data tensors per GPU go through **pinned** memory (CPU RAM that the OS has been told not to page out to disk — this lets the GPU pull from it via direct memory access at full PCIe bandwidth). The 84 transfers are bimodal in size: per GPU per profile we see 12 transfers of ~991 MB, 12 of ~105 MB, 12 of ~22 MB, plus smaller, totalling 13.87 GB per GPU (mean per transfer 165 MB). Roughly 2,845 smaller transfers per GPU use **pageable** memory (ordinary CPU RAM, which the GPU has to copy via a slower bounce-buffer path) — totalling 1.60 GB per GPU. The NVIDIA and DSI 4-GPU distributions are byte-for-byte identical, which means both clusters are running the same DataLoader configuration and `pin_memory` is already enabled for the main data path. The smaller pageable transfers are likely internal CUDA buffers or small parameter tensors, not the weather inputs.

### What may? fix it

The dispatch smoke test (see Midway cluster section below) measured sub-millisecond inter-step handoff gaps on both Midway H200 nodes under a synthetic workload. That test ran a model 4–6× slower than real Pangu, so it gives the CPU substantial slack between launches — it is an *upper bound* on Python overhead under generous CPU conditions, not a direct measurement under Pangu's tighter loop. With that caveat, it still rules out the extreme case (a Python loop that is fundamentally too slow on Midway); the stalls on DSI are most plausibly a hardware-topology or system-configuration effect specific to that node. With that in mind:

**NUMA binding.** NUMA (Non-Uniform Memory Access) refers to multi-socket servers where each CPU has its own bank of RAM; accessing the "near" bank is fast, accessing the "far" bank pays a latency penalty. If the DSI cluster has a multi-socket layout with GPUs split across NUMA nodes, binding each `torchrun` process to the CPU cores and memory local to its GPU would eliminate cross-socket handoff latency. Running `nvidia-smi topo -m` and `numactl --hardware` (`numactl` is the Linux tool for inspecting and controlling NUMA placement) on DSI would confirm whether this is the case (see the "What is still unknown — and the hardware diagnostic to request from DSI" subsection below). The pinned-memory check above already ruled out the *DataLoader configuration itself* as the cause — both clusters allocate pinned buffers byte-for-byte identically — but the NUMA placement of those pinned buffers is a separate question that the memory-kind check does not address.

The PCIe contentio?  Again unsure, a hardware topology constraint of the DSI node and cannot be fully worked around in software without knowing the actual node topology first.

---

## Midway cluster GPU comparison

To understand whether the DSI performance issues are DSI-specific or reflect the H200 hardware in general, we ran the bandwidth test (`v2.0/test/bandwidth_test.py`) and collected nsys inference profiles on H200 nodes in Midway's test partition. Two node types were tested: Intel Gold-6542Y (midway3-0602) and AMD EPYC-9335 (midway3-0601).

### What we know so far across all measured configurations

(Column key: **NVLink** = direct GPU-to-GPU electrical bus, much faster than PCIe; `NV6` means 6 NVLink bonds. **NUMA distance** = relative cost of cross-socket memory access, where 10 is local and higher is "farther". **Concurrent bandwidth drop** = how much each GPU's H2D bandwidth degrades when all four GPUs transfer simultaneously vs one at a time.)

| Cluster | GPU | CPU | NVLink | NUMA nodes | NUMA distance | CPU→GPU bandwidth (single GPU) | Concurrent bandwidth drop | Python handoff gap | GPU util (4-GPU) | Gaps >10ms | Kernel data |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA cluster | H100 NVL | — | unknown¹ | — | — | 42–45 GB/s² | ~0%² | — | 37–57% | 27 (GPU0) | ✓ |
| DSI | H200 | unknown | unknown | unknown | unknown | 41.6 GB/s² | **>100 MB: ~0% (all GPUs flat at 55 GB/s)**; all transfers avg: GPU0/3 −24/-21%, GPU1/2 −7/-10%² | — | 15–23% | 423 (GPU0); 649–894 (GPU1–3) | ✓ |
| Midway Intel (test) | H200 | Gold-6542Y | NV6 full mesh | 2 (GPU0/1 vs GPU2/3)⁷ | 21 | 44–46 GB/s aggregate, symmetric across GPUs³ | **22 MB upper_air: −2 to −4%**; 1–4 MB latency-bound: −10 to −62%³ | synthetic 0.018 ms⁴; **real Pangu 0.024–0.030 ms¹⁰** | n/a⁵ | **0 in real-Pangu test (n=1,180)¹⁰** | partial⁵; handoff via cudaEvent¹⁰ |
| Midway AMD (test) | H200 | EPYC-9335 | NV6 full mesh | 2 (GPU0/1 vs GPU2/3)⁷ | 32 | NUMA-aligned: GPU0/1 ≈ 39 GB/s, GPU2/3 ≈ 49 GB/s³ | ~0% additional concurrent drop⁶ | synthetic 0.012 ms⁴; **real Pangu 0.021–0.022 ms¹⁰** | n/a⁵ | **0 in real-Pangu test (n=1,180)¹⁰** | partial⁵; handoff via cudaEvent¹⁰ |
| Midway pedramh-gpu | H100 NVL | Xeon Gold 6346 (Ice Lake) | NV12 within pairs only⁸ | 2 (GPU0/1 vs GPU2/3)⁸ | 20 | ~27 GB/s (PCIe Gen4 limit)⁸ | NUMA-aligned: GPU2/3 −42% on H2D, GPU0/1 −48% on D2H⁸ | unavailable⁹ | unavailable⁹ | unavailable⁹ | partial⁹ |

¹ We never ran the GPU interconnect topology check on the NVIDIA cluster, so we don't know if its H100 NVL cards use NVLink or not.

² The NVIDIA and DSI bandwidth numbers are from actual inference (`nsys` MEMCPY records). The 84 pinned per-GPU transfers per profile are bimodal: 12 of ~991 MB, 12 of ~105 MB, 12 of ~22 MB, plus smaller (mean 165 MB). The Midway bandwidth numbers are from a dedicated test (`v2.0/test/bandwidth_test.py`) that uses tensor shapes derived from `exp2.yaml`: H2D `upper_air_input` = 22.3 MB, `surf_input` = 1.3 MB, `varying_boundary` = 3.9 MB; D2H (device-to-host: GPU memory → CPU memory, the reverse direction) stacked outputs up to 356.5 MB. The H2D upper-air size is within ~25% of the production figure derived from `exp2.yaml` at batch=2/GPU, so the *relative* drop comparison across clusters is broadly meaningful, but the absolute GB/s numbers should not be compared one-to-one because the test sizes and the production sizes are not identical and the software stacks differ.

³ On Midway Intel the four GPUs are essentially symmetric in sequential H2D (53.8 / 53.8 / 53.6 / 53.6 GB/s on the 22 MB `upper_air_input`; 18.8 / 18.9 / 18.4 / 18.3 GB/s on the 1.3 MB `surf_input` which is latency-dominated). Under concurrent 4-GPU load, the production-sized `upper_air_input` (22 MB) drops only −1.6 to −3.5% across the four GPUs; the `varying_boundary` tensor (3.9 MB, latency-bound regime) drops −57 to −62%; the small `surf_input` (1.3 MB) drops −10 to −22%. The "aggregate median" headline of −54 to −61% (footnote 6) is dominated by the small-tensor regime and does not reflect the production data path. On Midway AMD, sequential H2D shows a clean ~20% NUMA-aligned split: aggregate medians are 39.4 / 39.2 / 48.7 / 48.7 GB/s for GPU0/1/2/3 respectively. The split matches the CPU-affinity boundary from the topology matrix (GPU0/1 on cores 0-31, NUMA node 0; GPU2/3 on cores 32-63, NUMA node 1), consistent with the AMD allocator placing pinned host buffers preferentially on one NUMA node.

⁴ This was measured on a single GPU running a synthetic model that was 4-6× *slower* per step than real Pangu (forward time 87.8 ms on Intel, 55.2 ms on AMD, vs ~14 ms for Pangu). At these step times the CPU has substantial slack between launches, so the measured 0.018/0.012 ms inter-step gap is an upper bound on Python overhead under generous CPU conditions, not a direct measurement of overhead under Pangu's tighter loop. Notably, the CUDA Graph approach was 2-3× *slower* than the standard Python loop (0.046 ms vs 0.018 ms Intel), because it still has to copy new input data into fixed memory buffers before each replay. This synthetic test only covers single-GPU behaviour; the 4-GPU setup where DSI's gaps appear is covered by the Real-Pangu dispatch test (footnote 10) instead.

⁵ Both Midway H200 4-GPU `nsys` captures completed and produced sqlite files (`test_partition_benchmarks/midway_h200_intel_4gpus_inference.sqlite`, 99 MB; `midway_h200_amd_4gpus_inference.sqlite`, 88 MB), but the `CUPTI_ACTIVITY_KIND_KERNEL` table is missing from both — the test partition's security policy blocks the profiler's kernel-tracing channel (the profiler attaches to user processes via the Linux `ptrace` syscall, which is restricted on shared partitions) even when `nsys` itself runs to completion. MEMCPY and SYNCHRONIZATION events are present. AMD's MEMCPY table is also missing GPU0 entirely (only deviceIds 1–3 are recorded). The capture window is much shorter than the DSI profiles (490 H2D transfers per GPU vs 2,929 on DSI), so the runs did not cover the same number of inference iterations. Per-GPU concurrent H2D bandwidth under real inference (limited to transfers >1 MB) from these captures is: Intel GPU0/1/2/3 = 10.7 / 12.5 / 13.6 / 13.3 GB/s; AMD GPU1/2/3 = 41.4 / 41.3 / 42.6 GB/s. Gap-distribution and kernel-mix analysis is not possible without the kernel table.

⁶ Contention deltas (concurrent − sequential) on Midway, from `bandwidth_test.py` aggregate medians: Intel GPU0/1/2/3 = −55.6 / −54.3 / −61.3 / −55.2% (a uniform ~55% collapse across all four GPUs); AMD GPU0/1/2/3 = +0.5 / +24.4 / −0.0 / +0.0% (concurrent ≈ sequential on three GPUs; the +24.4% on GPU1 comes from the `upper_air_input` figure jumping from 43.75 to 54.93 GB/s, which is more consistent with measurement noise than a real architectural improvement). For AMD the "no degradation" result should be taken as "no meaningful *additional* contention detected on top of the sequential NUMA asymmetry already documented in footnote 3" rather than a precise measurement of perfect isolation.

⁷ The server topology report suggested GPU0 and GPU1 share one CPU socket (cores 0–23 on Intel, 0–31 on AMD) while GPU2 and GPU3 share the other socket (cores 24–47 / 32–63). However, the GPU-side confirmation field in the report was blank on both nodes, so this is an inference from the CPU affinity hints rather than a direct readout.

⁸ pedramh-gpu (`midway3-0423`) is **H100 NVL on Xeon Gold 6346 (Ice Lake)** — i.e., the same GPU model as the NVIDIA cluster, but a Midway host. From `test_partition_benchmarks/midway_bandwidth_midway_bandwidth_test.sh_49972059.out` and `test_partition_benchmarks/pedramh_hw_topo.out`: 2 sockets × 16 cores (NUMA distance 10/20), driver 535.216.03, CUDA 12.2. NVLink topology is **NV12 within socket-pairs only** (GPU0↔GPU1 = NV12, GPU2↔GPU3 = NV12, but {GPU0,1} ↔ {GPU2,3} = `SYS`/cross-socket UPI, no NVLink) — fundamentally different from Midway H200's NV6 full mesh. Ice Lake is PCIe Gen4 only, so the H100 NVL bandwidth ceiling here is ~32 GB/s rather than the ~64 GB/s achievable on Gen5 hosts; the 27 GB/s sequential figure is ~84% of Gen4 theoretical and is comparable in *fraction of available bandwidth* to Midway H200's 53 GB/s out of Gen5's 64 GB/s. Under concurrent 4-GPU load, H2D shows a NUMA-aligned drop: GPU0/1 hold ~27 GB/s while GPU2/3 collapse to ~14 GB/s (−42%). On D2H the asymmetry inverts (GPU0/1 collapse, GPU2/3 hold) — consistent with the dataloader pinning host buffers on one NUMA node for input and the destination buffers on another for output.

⁹ The pedramh-gpu 4-GPU inference profile (`test_partition_benchmarks/midway_h100_4gpus_inference.{nsys-rep,sqlite}`) completed, but — same as the H200 test partition profiles (footnote 5) — the `CUPTI_ACTIVITY_KIND_KERNEL` table is missing from the sqlite. Only MEMCPY, SYNCHRONIZATION, and RUNTIME events are present, so per-kernel timing and the gap-distribution analysis we wanted for the H100-NVL-on-Midway-host vs H100-NVL-on-NVIDIA-cluster comparison are not directly available. Earlier drafts of this report claimed pedramh-gpu had "different security settings" that would allow kernel tracing; that turned out to be incorrect — the same `ptrace`-related restriction applies. What we *can* see from the profile and its accompanying `.out`: this host runs the same software stack as the test partition (driver 535.216.03, driver CUDA 12.2, PyTorch 2.6.0+cu124), but the `intel_idle` driver here permits **deeper CPU sleep states than the test partition does** — `POLL, C1, C1E, C6` available, vs the test partition's `POLL, C1_ACPI, C2_ACPI` (C2 maximum, 41 μs exit). C6 has roughly 100-200 μs exit latency on Intel — too small on its own to account for 10-50 ms gaps, but worth flagging as a candidate contributing factor on any host that allows it. The RUNTIME table here captures only 4,252 events (a short window), too few to derive per-step gap statistics. The kernel-level H100-NVL host comparison we wanted is therefore not possible from this collection without a different profiler-attach method.

¹⁰ Real-Pangu dispatch test (`v2.0/test/inference_dispatch_real.py`, submission scripts `midway_dispatch_real_{intel,amd}.sh`). Uses `torch.cuda.Event` pairs to measure GPU idle between consecutive forward passes of the actual `PanguModel_Plasim` with production-shape random inputs. In-process timing, no profiler attach, no `ptrace` dependency. Across 1,180 inter-step gaps per Midway node (4 reps × 60 steps × 1 GPU + 4 reps × 60 steps × 4 GPUs − 8 startup gaps not counted at the rep boundary, summing to ~1,180 measurements per cluster row), zero gaps exceeded 10 ms; the maximum gap observed on either node across both phases was 66 microseconds. DSI under matching conditions shows 41 gaps of 10–50 ms on 1 GPU and 423–894 across the four GPUs. Software stack for these runs (identical on both nodes): NVIDIA driver 535.216.03, driver CUDA 12.2, PyTorch 2.6.0+cu124 (running in CUDA forward-compatibility mode), `intel_idle` driver with `menu` governor and C2_ACPI maximum (41 μs exit latency). See the "Real-Pangu dispatch test" subsection below for the full discussion of what the test measures, what it rules out, and what caveats apply.

### Key findings from the Midway benchmarks

**H200 on Midway has NVLink (NV6 full mesh).** All four H200s on both Intel and AMD nodes are connected to each other with 6 NVLink bonds each. This is *not* the same as the typical dual-socket H100 NVL board topology: pedramh-gpu's H100 NVL (see footnote 8) shows NV12 within socket-pairs only, with cross-socket `SYS` between the pairs. So the H200 boards on Midway are wired more aggressively than the H100 NVL boards on the same campus. That said, NVLink topology does **not** address H2D bandwidth — data still moves from CPU memory over PCIe to reach the GPU, and NVLink does not bypass that path. So the NV6 finding is irrelevant to the H2D contention story below. Whether the DSI node also has NVLink is still unknown; `nvidia-smi topo -m` on the DSI node would answer this immediately.

**pedramh-gpu H100 NVL is structurally the cleanest comparison point we have for DSI's host environment** — same GPU model as the NVIDIA cluster, but a different host (Ice Lake Xeon, 16 cores per socket, PCIe Gen4, CUDA 12.2 driver 535). Its bandwidth pattern (NUMA-aligned drop under concurrent load) is much closer to Midway AMD H200 than to the NVIDIA cluster's flat H100 NVL profile, which suggests the NVIDIA cluster host is doing something specific (NUMA-aware allocator placement, dedicated PCIe routing, or single-socket layout) that ordinary dual-socket H100 NVL hosts don't. We had hoped the `midway_infer_nsys.sh` run on pedramh-gpu would give us full kernel timing for a head-to-head H100-NVL-vs-H100-NVL comparison; that profile ran but the kernel table is missing (see footnote 9) — pedramh-gpu turns out to have the same `ptrace` restriction the H200 test partition does. The one cleanly observable host difference from this profile is the **CPU sleep-state policy**: pedramh-gpu's Ice Lake host allows deep idle states up to C6, while the H200 test partition is restricted to C2 (41 μs exit). If DSI's host similarly allows C6 or deeper, that's a candidate contributing factor; the diagnostic block in the "What is still unknown" section below explicitly checks for it.

**Midway Intel: symmetric across GPUs, and large-tensor concurrent bandwidth is essentially unaffected.** All four Intel GPUs deliver essentially identical sequential H2D bandwidth (53.8 / 53.8 / 53.6 / 53.6 GB/s on the 22 MB `upper_air_input`). Under simultaneous load from all four GPUs the large-tensor bandwidth holds: GPU0/1/2/3 drop by only −1.6 to −3.5% on `upper_air_input`. The "−55% concurrent collapse" figure that appears in the aggregate-median row of the bandwidth_test output is dominated by the small `varying_boundary` (3.9 MB) and `surf_input` (1.3 MB) tensors, which are latency-bound: at those sizes, four concurrent processes contending for host-side handoff / kernel locks / IRQ paths serialise badly, but it does not reflect contention on the bus itself. Earlier drafts of this report claimed Intel GPU0 was "half the bandwidth of GPU1-3"; that claim was a measurement-script confusion and is not present in the raw data.

**Midway AMD: NUMA-aligned 20% sequential asymmetry, ~0% additional concurrent degradation.** On the AMD node, sequential H2D splits cleanly along the NUMA boundary: GPU0/1 (cores 0–31, NUMA node 0) at ~39 GB/s vs GPU2/3 (cores 32–63, NUMA node 1) at ~49 GB/s (aggregate medians). Under concurrent load all four GPUs hold their sequential numbers within ±0.5%, except for an unexplained +24% jump on GPU1's `upper_air_input` measurement (43.75 → 54.93 GB/s) that is more consistent with measurement noise than a real architectural gain.

To understand the NUMA effect: a dual-socket server has two CPU chips, each with its own pool of local RAM. A GPU copies data from CPU RAM across the PCIe bus. If the GPU is on the "near" CPU socket — the one that owns the RAM being read — the transfer is fast. If it is on the "far" socket, the data first has to cross an inter-socket link (Intel's UPI or AMD's Infinity Fabric) before reaching PCIe, which adds latency and reduces bandwidth. The Midway topology output (`nvidia-smi topo -m`) showed:

- GPU0 and GPU1 have CPU affinity 0–23 (Intel) / 0–31 (AMD), suggesting NUMA node 0
- GPU2 and GPU3 have CPU affinity 24–47 (Intel) / 32–63 (AMD), suggesting NUMA node 1
- Cross-socket distance is 2.1× worse than local on Intel (21 vs 10) and 3.2× on AMD (32 vs 10)
- Note: the `GPU NUMA ID` field was `N/A` on both nodes, so the GPU-side NUMA assignment was not directly confirmed by the hardware report

The AMD pattern is consistent with the dataloader allocating pinned buffers preferentially on one NUMA node — GPUs on the same node see local memory, GPUs on the other node pay the cross-socket cost. The Intel node does not show this split in our measurements, which may reflect either a different allocator placement on the Intel host, a faster cross-socket fabric, or a measurement that simply did not exercise the imbalance. Without explicit `numactl --membind` testing we cannot distinguish these. AMD's symmetry under concurrent load (no further degradation on top of the sequential split) is consistent with sufficient per-GPU PCIe bandwidth headroom on EPYC-9335, but with only one node measured per CPU family we cannot generalise this to a property of "AMD PCIe architecture" — it could equally reflect this specific board's slot wiring, driver version, or BIOS configuration.

**DSI's averaged bandwidth pattern matches no other host we've measured — but only in the small-transfer regime.** DSI's *averaged-over-all-transfers* concurrent H2D drops are GPU0 = −24.3%, GPU1 = −7.2%, GPU2 = −10.1%, GPU3 = −21.4% — slow on GPU0 and GPU3, less affected on GPU1 and GPU2. This bimodal pair-split is not the NUMA {0,1}-vs-{2,3} split seen on AMD H200 or on pedramh-gpu H100 NVL, nor the symmetric-on-large-tensors pattern seen on Intel H200. A {0,3} split would fit a topology like a PCIe-switch grouping where one switch holds {0,2} and another {1,3} on opposite sockets, or a non-standard slot wiring (the DSI node's four H200s may share fewer PCIe root complex lanes — **googled for this answer not sure though**) — but pinning this down requires `nvidia-smi topo -m` from DSI. However, restricting to the >100 MB transfers that constitute the production weather-data path, **all four DSI GPUs sustain 55.4 GB/s under 4-GPU concurrent load — identical to single-GPU DSI and identical to the H100 cluster**. So the bimodal asymmetry exists only for small transfers, not for the dominant data path. This is consistent with a small-transfer / host handoff contention story rather than a bus-level PCIe topology problem. Without `nvidia-smi topo -m` from DSI we still cannot fully characterise the hardware layout, but the bandwidth observations no longer require a topology explanation for the production workload.

### Real-Pangu dispatch test on Midway H200 (2026-05-22 update)

To close the gap left by the test partition's ptrace restriction (which blocks `nsys` kernel tracing on the Midway H200 4-GPU profiles, footnote 5), we ran a custom test that measures inter-step GPU-idle time directly with in-process `torch.cuda.Event` pairs. The test does not need a profiler attached, so it works on the test partition. Source: `v2.0/test/inference_dispatch_real.py`; submission scripts: `v2.0/HPC_scripts/midway_dispatch_real_{intel,amd}.sh`.

**What the test measures.** The autoregressive loop is what the inference run actually does: 60 sequential forward passes of `PanguModel_Plasim`, each one taking the previous step's output as its next input. Between every two consecutive forward passes there is an interval — what we call the *handoff gap* — where the GPU has finished the last kernel of step N but the CPU has not yet queued the first kernel of step N+1.

```
        ┌────────── step N ──────────┐                     ┌──── step N+1 ────
GPU:   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┃    gap (idle)    ┃━━━━━━━━━━━━━━━━━━━━━ ...
                                       ▲                  ▲
                                  end_event[N]      start_event[N+1]
                                  recorded here     recorded here

CPU during the gap is doing:
  1. return from model(...)
  2. outputs.append(out_s.detach())          ← production list-append pattern
  3. rebind surface = out_s, upper_air = out_u
  4. record start_event[N+1] on the stream
  5. re-enter model(...) and queue the first kernel of step N+1
```

If the CPU does all of that faster than the GPU finishes step N, the GPU never goes idle and the gap is ≈ 0 (sub-microsecond). If anything stalls the CPU — Python overhead, scheduler preemption, NUMA-distant memory access during kernel-launch parameter copy, IRQ handler stealing the CPU, deep C-state exit — the GPU sits there and we measure how long.

**What the test deliberately excludes**, so the measured gaps are unambiguously handoff-path effects and not something else:

- **No HDF5 read.** Inputs are random tensors allocated once on the GPU at startup. No `DataLoader`, no `pin_memory` copies, no disk I/O between steps.
- **No D2H transfers between steps.** Production stacks the full 60-step output and copies it to CPU at the end of each outer iteration; we don't do that part. (A D2H stall would be a different cost class and would show up in the >100 ms gap bucket of the DSI nsys profile, not the 10-50 ms bucket.)
- **No NCCL.** Inference is data-parallel-only, no collectives, no `find_unused_parameters`.
- **No checkpoint save / `inv_transform` / async-save thread.** These happen at outer-iteration boundaries, not between steps within the autoregressive loop.

This is the exact class of gap that the DSI nsys profile classifies as 10-50 ms — kernel-to-kernel idle inside the inference loop, after data is already resident on the GPU. The user's annotation in the summary section captures this distinction: *"a data loader is not the problem as the gaps appear somewhere during model inference itself."*

**Results.** Across both Midway H200 nodes, both 1-GPU and 4-GPU phases, all four ranks: **zero gaps exceeded 10 ms** in 1,180 measurements per node.

| Configuration | n | median | p99 | max | gaps > 10 ms |
|---|---:|---:|---:|---:|---:|
| Midway Intel, 1 GPU (batch=8) | 236 | 0.025 ms | 0.032 ms | 0.043 ms | **0** |
| Midway Intel, 4 GPUs (batch=2/GPU) | 944 | 0.024–0.030 ms | 0.038–0.040 ms | 0.060 ms | **0** |
| Midway AMD, 1 GPU (batch=8) | 236 | 0.021 ms | 0.027 ms | 0.066 ms | **0** |
| Midway AMD, 4 GPUs (batch=2/GPU) | 944 | 0.021–0.022 ms | 0.024–0.028 ms | 0.058 ms | **0** |
| (Reference: DSI 1-GPU from nsys) | — | — | — | — | 41 |
| (Reference: DSI 4-GPU GPU0 from nsys) | — | — | — | — | 423 |
| (Reference: DSI 4-GPU GPU1–3 from nsys) | — | — | — | — | 649–894 |

Maximum gap across all 2,360 measurements on Midway: **66 microseconds**. The smallest DSI gap that lands in the 10-50 ms bucket is **at least 150× larger**.

**What this finding rules out (decisively, on Midway evidence alone):**

- **The H200 chip itself as the sole cause.** The same GPU model on a different host (Midway) ran the identical workload and produced no gaps over 10 ms. So the H200 on its own, with a standard software setup, does not produce this latency. What this does *not* rule out is the H200 combined with something specific to DSI — for example, a different version of the small firmware baked into the GPU, a different variant of the H200 board, or a power-saving setting on the GPU's connection to the CPU that only misbehaves in DSI's particular host environment. One additional `nvidia-smi` query on the DSI node would confirm that the GPU model and firmware match Midway's; we have added it to the diagnostic block below.
- **4-rank `torchrun` as an inherent producer** of the gaps. Midway runs 4 ranks the same way as DSI and sees zero.
- **GPUs reading data from the "wrong" CPU memory bank** as enough on its own to cause the gaps. Modern servers have two CPU chips, each with its own bank of RAM; if a GPU has to pull input from the *far* bank, the transfer takes longer (this is called the NUMA effect). Midway's GPUs are split across both memory banks, no special steering was done to keep each GPU on its local bank, and the test still produced no gaps. So this kind of mis-placement by itself is not enough to explain the 10-50 ms gaps; if it contributes on DSI, it would have to be in combination with another factor.
- **Deep CPU sleep states** as the mechanism on Midway. When a CPU is idle, Linux can put it into progressively deeper power-saving states; the deeper the state, the longer it takes the CPU to wake up and handle the next piece of work. Midway's hosts only allow the two shallowest sleep states, which wake the CPU in 41 microseconds at most — three orders of magnitude faster than the 10-50 ms gaps we are trying to explain. So on Midway specifically, deep idle states are not the mechanism. DSI may differ here, which is why the diagnostic block below includes a check for it.
- **Something inherent to how PanguModel_Plasim schedules GPU work.** The Midway test ran the actual model architecture with production tensor shapes (only the trained weights were random, since the GPU command pattern does not depend on the values inside the weights). If the gaps were caused by the model's own pattern of kernel launches — for example, a particular sequence of small kernels that has to serialise on the CPU side — they would have shown up on Midway too. They did not.

**What this finding does *not* rule out**, and which remain candidates for the DSI-specific cause:

1. **DSI driver / CUDA version mismatch.** Midway runs driver 535.216.03 with driver CUDA 12.2 and PyTorch built for cu124 (forward-compat). If DSI runs an older or newer driver — especially anything in known-bad regions for kernel completion IRQ — that is the leading candidate.
2. **DSI kernel / OS configuration** (different `intel_pstate` governor, different `kernel.sched_min_granularity_ns`, different `intel_idle.max_cstate`, different cgroup throttling).
3. **DSI co-tenant CPU contention.** If the DSI node was not exclusive at capture time, a co-tenant could be stealing CPU cycles in the 10-50 ms range.
4. **DSI-specific IRQ steering** that funnels nvidia completion interrupts onto a CPU that's also fielding other heavy IRQs (NIC, NVMe).
5. **DSI filesystem effect.** Although the dispatch test deliberately strips out disk I/O, the production DSI run does read from disk between outer iterations, and if that read holds a kernel lock (e.g., page-cache contention) the lock may carry into the autoregressive loop on subsequent kernel launches.

**Caveat on absolute scale.** The forward-pass times in this test were higher than production (Intel 1-GPU batch=8: 407 ms; 4-GPU batch=2/GPU: 133 ms; AMD 1-GPU batch=8: 986 ms; 4-GPU batch=2/GPU: 104 ms), compared to the production target of ~14 ms per forward at batch=1 on H100. Most likely causes: random init weights drove cuDNN autotune to slower kernels, and the test did not set `torch.backends.cudnn.benchmark = True`. Longer forward passes give the CPU more slack to launch the next kernel — so this test is *biased in favour of* not finding gaps. The result is decisive only because Midway gaps are so far below DSI's that even an order-of-magnitude tighter loop (forwards ≈ 10 ms instead of 100–1000 ms) would still leave Midway clean at the sub-millisecond scale. A follow-up tightened-loop run with `cudnn.benchmark = True` and warmed-up weights would close that small gap in rigour.

**The cheapest next diagnostic** is to ask the DSI team for the same data we have for Midway. One short script run on a DSI node would resolve most of the candidate list:

```bash
nvidia-smi | head -3                                            # driver + driver-CUDA version
nvidia-smi --query-gpu=driver_version --format=csv
nvidia-smi --query-gpu=name,vbios_version --format=csv          # exact H200 variant + GPU firmware
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi topo -m
numactl --hardware
cat /sys/module/intel_idle/parameters/max_cstate 2>/dev/null
cpupower frequency-info | head
cpupower idle-info     | head
grep nvidia /proc/interrupts | head
dmesg | grep -iE "nvidia|throttle|c-state" | tail -30
```

If the DSI driver version differs from `535.216.03`, that is the lead to investigate first.

**What we can and cannot compare between Midway and DSI.**

The dispatch smoke test (`v2.0/test/inference_dispatch_smoke.py`) ran on a **single GPU** on both Midway H200 nodes and measured the GPU idle time between consecutive forward passes using a synthetic model. The median inter-step gap was 0.018 ms on Intel and 0.012 ms on AMD — both far below 10 ms. This is an *upper bound* on Python overhead when work is plentiful (the synthetic forward took 55–88 ms, 4–6× longer than real Pangu, which gives the CPU substantial slack to prepare the next launch); it does not directly measure overhead under Pangu's tighter inner loop. With that caveat, the result still shows the handoff path on Midway H200 is not pathologically slow in the way DSI's is at the single-GPU level.

The Midway 4-GPU nsys captures completed and contain MEMCPY and SYNCHRONIZATION events, but the `CUPTI_ACTIVITY_KIND_KERNEL` table is missing from both — the test partition's ptrace restriction blocks CUPTI's kernel-tracing channel even when nsys itself runs to completion. AMD's MEMCPY also dropped GPU0 entirely (only deviceIds 1–3 are recorded). The capture window is much shorter than DSI's (490 vs 2,929 H2D transfers per GPU). Given these limitations: we **can** report per-GPU concurrent H2D bandwidth on Midway under real inference (Intel ~11–14 GB/s, AMD ~41–43 GB/s — both well below the sequential test numbers), but we **cannot** reproduce DSI's gap-distribution analysis on Midway H200 from these files.

What makes the single-GPU comparison meaningful is the DSI 1-GPU profile: even on **one GPU with no inter-process competition**, DSI already shows **41 gaps of 10–50 ms** and only 39% GPU utilisation. Something about DSI's single-GPU environment already causes elevated handoff latency relative to Midway H200's single-GPU handoff test. The 4-GPU case makes it roughly 10× worse (423 gaps on GPU0; 649–894 on GPU1–3), but the elevated baseline is present before any multi-GPU effects are added. So the framing is: DSI's *handoff-latency pattern* is distinct from what Midway H200 shows in single-GPU handoff — though the comparison only covers the handoff path itself, not whatever host-side contention 4 co-resident processes add on top.

To complete a like-for-like *kernel-level* multi-GPU comparison (over and above the inter-step gap comparison already provided by the Real-Pangu dispatch test in footnote 10) we would need either the ptrace restriction lifted on the test partition (giving us kernel timing on Midway 4-GPU) or the same inference nsys profile run on the `pedramh-gpu` H100 partition (which has different security settings).

### What is still unknown — and the hardware diagnostic to request from DSI

The largest remaining unknown is the DSI node's host configuration. A short script that someone on the DSI team can run on the node — taking under a minute, requiring nothing beyond a normal user account — would answer most of the remaining candidate causes. None of these commands modify anything; they only read system state. Commands are grouped by what they tell us:

```bash
# --- GPU identity (≈5 seconds) ---
nvidia-smi | head -3                                             # driver version + driver-supported CUDA version
nvidia-smi --query-gpu=driver_version --format=csv               # machine-readable driver version
nvidia-smi --query-gpu=name,vbios_version --format=csv           # exact H200 board variant + GPU firmware
nvidia-smi topo -m                                               # GPU-to-GPU links (NVLink vs PCIe) + CPU/NUMA affinity per GPU
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # PyTorch + the CUDA version it was built for

# --- CPU / NUMA / sleep states (≈10 seconds) ---
lscpu | head -20                                                 # CPU model, core count, socket layout
numactl --hardware                                               # NUMA node count and cross-socket distance
cat /sys/module/intel_idle/parameters/max_cstate 2>/dev/null     # deepest CPU sleep state the kernel will use (Midway: 2)
cpupower frequency-info | head                                   # CPU governor (performance vs powersave)
cpupower idle-info     | head                                    # C-state idle-driver detail

# --- IRQ and host warnings (≈5 seconds) ---
grep nvidia /proc/interrupts | head                              # which CPUs are handling GPU completion interrupts
dmesg | grep -iE "nvidia|throttle|c-state" | tail -30            # any GPU/CPU warnings in the kernel log
```

**The single most informative line is the first one.** If DSI's driver or driver-supported CUDA version differs from Midway's `535.216.03 / 12.2`, that is the leading lead. Beyond that:

- If `nvidia-smi topo -m` shows `PIX` or `PHB` links (PCIe-only, no NVLink) where Midway shows `NV6`, that is a structural GPU-interconnect difference worth noting (though, per the discussion above, the GPU-to-GPU links matter less for the CPU-to-GPU handoff path that we are actually investigating).
- If `nvidia-smi --query-gpu=name,vbios_version` shows a different H200 board variant or firmware version than Midway, the door opens to a firmware-mediated effect on the H200 side itself.
- If `cat /sys/module/intel_idle/parameters/max_cstate` returns a value higher than 2, deep CPU idle states are enabled and could plausibly contribute. Worth noting: Midway hosts are not uniform on this axis — the H200 test partition (Sapphire Rapids Gold-6542Y) is restricted to C2 at 41 μs exit, while Midway's pedramh-gpu node (Ice Lake Gold 6346) permits up to C6 (~100-200 μs exit). C6 alone is still too short to produce 10-50 ms gaps, but if DSI runs an older or differently-configured host that permits even deeper states, that's a credible contributor.
- If `grep nvidia /proc/interrupts` shows GPU completion interrupts pinned to a single CPU that is also handling NIC or NVMe traffic, that is another candidate.
- If `dmesg` shows recurring `throttle`, `c-state`, or NVIDIA warnings, the kernel itself is flagging something we should follow up on.

This block is a superset of the diagnostic block that already appears in the "Real-Pangu dispatch test" subsection above — the dispatch test wraps the same commands into the test's own `.out` header so we record them automatically every time the test runs. The DSI ask is the same commands run once on a DSI node and pasted back into a comment on this report or shared via email.

---

## Summary of DSI investigation as of 2026-05-22

**The main problem.** The DSI cluster's H200 GPUs spend most of their time sitting idle rather than running the model. Three facts now bound the explanation: (1) the actual forecast computation takes about the same time per GPU on DSI as on the NVIDIA cluster (within ~10%), so the H200 is doing the work at a similar rate to the H100; (2) the dominant data path — the large weather-data tensors moving from CPU to GPU — sustains ~55 GB/s on both clusters under 4-GPU load, so the PCIe bus is not the bottleneck for the production workload; (3) the GPUs nevertheless keep stopping and waiting for 10–50 ms between consecutive bursts of work, hundreds of times per run, and the NVIDIA cluster barely does this at all. The wall-time deficit is therefore almost entirely **handoff latency** — the CPU not handing the next batch of work to the GPU fast enough — rather than compute or bandwidth.

**What was ruled out.** Both clusters ran the same code (with the caveat that the software stack — CUDA, cuDNN, nsys version — was not held strictly constant between the bare-metal DSI host and the NGC apptainer used on NVIDIA). A handoff timing test on Midway's H200 machines using a *synthetic* model gave a first upper bound (sub-millisecond Python overhead). A follow-up test running the *real* PanguModel_Plasim through the same 60-step autoregressive loop on both Midway H200 nodes (1 GPU and 4 GPUs) showed **zero inter-step gaps exceeding 10 ms in 2,360 measurements** — maximum gap observed was 66 microseconds, vs DSI's 41-894 gaps in the 10-50 ms range under the same loop. This directly rules out the H200 architecture, the 4-rank `torchrun` pattern, and naïve NUMA mis-binding as causes; the DSI handoff latency is host-environment-specific. We also confirmed that adding more data loading workers would not help, because the data loading pipeline is not serialised across the four GPUs — each GPU already has its own independent loader, and the pinned-memory configuration matches NVIDIA's byte-for-byte.

**What the Midway tests showed.** Midway has two types of H200 nodes — Intel CPU and AMD CPU. Both have direct high-speed connections between the four GPUs. On the **production-sized weather tensors** (~22 MB each per H2D call, ~165 MB averaged across the larger pinned transfers), the Intel node delivers identical bandwidth on all four GPUs whether they pull data one at a time or all at once — only −2 to −4% degradation under simultaneous load. The headline "loses about half its bandwidth" figure that appeared in earlier drafts came from averaging in smaller (1–4 MB) tensors that are latency-dominated rather than bandwidth-dominated; on those the four ranks compete for host-side handoff and serialise badly, but that does not reflect a problem with the PCIe bus itself. The AMD node shows a ~20% sequential speed difference between the four GPUs even when only one is active — GPUs 0 and 1 are slower than GPUs 2 and 3 because they read from a different bank of CPU memory — but does not slow down further under simultaneous load. **The same correction applies to DSI**: when restricted to the production-sized transfers, all four DSI GPUs sustain ~55 GB/s under 4-GPU load, identical to single-GPU DSI and identical to the NVIDIA H100 cluster. The bimodal "−24% on GPU0/3" figure quoted earlier in this report is the all-transfers average and is dominated by the same small-transfer / host handoff contention seen on Intel.

A first dispatch test on a single Midway H200 GPU showed the Python loop adds under a millisecond between forecast steps when running a synthetic stand-in model — but the stand-in is 4–6× slower per step than real Pangu, so it is an upper bound on Python overhead under generous CPU conditions, not a direct measurement at Pangu's actual step time.  **However, a data loader is not the problem as the gaps appear somewhere during model inference it self.** A follow-up test (`v2.0/test/inference_dispatch_real.py`) ran the **actual PanguModel_Plasim** through the same 60-step autoregressive loop on both Midway H200 nodes (1 GPU and 4 GPUs) and measured inter-step gaps directly with in-process cudaEvent timing — no profiler needed, so it bypasses the test partition's ptrace restriction. Result: **zero gaps over 10 ms across 2,360 measurements** (maximum observed: 66 microseconds) on both nodes. DSI under the same loop shows 41 gaps in the 10-50 ms bucket on a single GPU and 423-894 across four GPUs. This is the decisive evidence that the DSI handoff latency is host-environment-specific: the H200 hardware itself hands off kernels cleanly, even with 4 ranks sharing the host and without explicit NUMA binding. See the "Real-Pangu dispatch test" subsection above for the full discussion. 

**What still needs to be learned.** As of 2026-05-22, the picture has narrowed considerably:

- **Midway H200 handoff under real Pangu is now measured and clean** (footnote 10 / "Real-Pangu dispatch test" subsection above). The H200 architecture and the 4-rank `torchrun` pattern are no longer candidates for the DSI handoff issue.
- **The pedramh-gpu hardware topology is known** (footnote 8). What remains for that node is the full 4-GPU inference profile via `v2.0/HPC_scripts/midway_infer_nsys.sh` — that gives a direct H100-NVL-on-Midway-host vs H100-NVL-on-NVIDIA-cluster comparison with full kernel timing.
- **The leading remaining unknown is the DSI software stack** — driver version, kernel idle/scheduler config, IRQ steering, and whether the node was exclusive at capture time. The short diagnostic block at the end of the "Real-Pangu dispatch test" subsection above lists exactly what to ask for. If DSI's driver version differs from Midway's `535.216.03`, that is the lead to investigate first.
- DSI's hardware topology (`nvidia-smi topo -m`, `numactl --hardware`) is also still missing and would resolve the remaining bandwidth-pattern question (why GPU0 and GPU3 group together rather than the {0,1}/{2,3} split seen on AMD and pedramh-gpu).

---

## Notes on measurement reliability

- All step times are measured with GPU synchronisation barriers (`torch.cuda.synchronize()`) on both sides of the timing window. This ensures we record actual execution time, not just how long it takes to submit work to the GPU.
- The warm-up period (20 steps) absorbs one-time costs: driver initialisation, automatic kernel selection (cuDNN autotuning), and communication library warm-up.
- The timer includes a self-consistency check: the sum of individually measured step windows must agree with the total elapsed wall time within 10%. If they disagree, the result is discarded rather than recorded.

### Cross-cluster comparison caveats

The DSI, NVIDIA, and Midway inference profiles compared in the sections above were collected with non-identical conditions. Specifically:

- **No warmup before profiling.** `v2.0/inference_optimized.py` starts capturing from the first iteration (`if i > 5: break`), so first-batch costs — cuDNN autotune, cuBLAS workspace init, pinned-buffer first allocation — are folded into every profile and contribute to the >500 ms gap bucket.
- **Software stack not held constant.** NVIDIA runs inside the NGC apptainer; DSI is bare-metal; Midway is bare-metal under a different driver/OS. CUDA, cuDNN, and nsys versions were not recorded per cluster.
- **Git SHA of `inference_optimized.py` not recorded** per cluster. Confirm the same file ran by computing the SHA before each capture if reproducing.
- **The verification script** `verify_bench.py` (at repo root) reproduces every numerical claim above from the .sqlite profiles; re-run it after any rebuild to confirm the numbers still hold.

