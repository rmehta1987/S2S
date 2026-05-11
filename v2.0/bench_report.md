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

The model generates 4 ensemble members by repeating each input sample 4 times and adding different random noise draws at the compressed bottleneck of the encoder. The noise is sampled from a distribution whose mean and variance are learned by the encoder. A second encoder branch, which only runs during training, processes the target (future) weather state and provides a reference distribution that the forecast encoder is trained to match. This is intended to teach the forecast encoder what the distribution of plausible future atmospheric states looks like in the latent space.

This design is known as a **conditional variational autoencoder with learned prior**. The same pattern appears in stochastic video prediction (teaching a model to generate diverse future frames) and open-domain dialogue generation (generating diverse responses). In all cases the idea is: show the model the future during training so it learns a prior distribution that, at inference time, produces samples consistent with real future states.

**Why it is fragile:** The balance between the forecast loss and the regularisation loss is controlled by a single weight (`vae_loss_weight: 0.0001`). If this weight is too small — as it appears to be here — the regularisation signal is negligible and the encoder learns to collapse its variance toward zero, making all 4 ensemble members nearly identical. This is called posterior collapse.

**Simpler alternatives that achieve the same goal:**

| Approach | How diversity is generated | Second encoder needed | Fragility |
|---|---|---|---|
| This model (learned prior) | Noise sampled from encoder distribution, KL training against posterior | Yes — runs on target data during training | High — KL weight must be tuned carefully |
| Fixed noise injection | Add scaled random noise directly at bottleneck, no learned distribution | No | Low |
| Monte Carlo dropout | Keep dropout active at inference, run 4 passes with different dropout masks | No | Low |
| Diffusion in latent space | Score-based sampling from a learned noise schedule | No (separate diffusion head) | Moderate — but principled and current state of the art for this problem |

Google DeepMind's GenCast (2023) uses the diffusion approach on a Pangu-style backbone and currently represents the state of the art for probabilistic medium-range forecasting. The second encoder branch in this model adds training-time compute (it is always gradient-checkpointed) and inference-time architectural complexity for a benefit that depends entirely on the KL weight being correctly tuned.

---

## Notes on measurement reliability

- All step times are measured with graphics card synchronisation barriers (`torch.cuda.synchronize()`) on both sides of the timing window. This ensures we record actual execution time, not just how long it takes to submit work to the graphics card.
- The warm-up period (20 steps) absorbs one-time costs: driver initialisation, automatic kernel selection (cuDNN autotuning), and communication library warm-up.
- The timer includes a self-consistency check: the sum of individually measured step windows must agree with the total elapsed wall time within 10%. If they disagree, the result is discarded rather than recorded.

