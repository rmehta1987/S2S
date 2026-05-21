#!/usr/bin/env python3
"""
PCIe host-to-device (H2D) bandwidth test for multi-GPU nodes.

Background
----------
When the CPU sends a tensor to a GPU, data travels over the PCIe bus from
CPU DRAM into GPU HBM. The rate of this transfer is the H2D bandwidth. On a
server with multiple GPUs, all cards share the same PCIe root complex (or
separate root complexes depending on the node topology). If they all transfer
simultaneously, they may compete for the same upstream bandwidth — this
competition is PCIe contention.

Additionally, modern servers have multiple CPU sockets (NUMA nodes), each
with their own local DRAM and their own set of GPUs attached to that socket's
PCIe lanes. A GPU transferring data from CPU DRAM that belongs to the *other*
socket has to cross a slow inter-socket link (UPI/QPI) first — this is a NUMA
penalty. It typically shows up as one or two GPUs being significantly slower
than the others in a concurrent test.

What this script measures
--------------------------
1. Sequential bandwidth  — each GPU is tested one at a time, with all others
   idle. This is the best-case H2D rate and is not affected by contention.
   It gives the per-GPU baseline.

2. Concurrent bandwidth  — all GPUs transfer simultaneously, each in its own
   OS process (mirroring torchrun's multi-process setup). This reveals whether
   bandwidth degrades under load, and whether the degradation is symmetric
   (PCIe contention) or asymmetric (NUMA penalty — only some GPUs slow down).

3. Contention delta  — the difference between concurrent and sequential
   bandwidth per GPU, as an absolute and percentage change. A large negative
   delta on GPU0 and GPU3 but not GPU1 and GPU2 (as seen on the DSI H200
   cluster) points to a NUMA topology issue rather than shared PCIe saturation.

Why pinned memory
-----------------
The test allocates the source tensor in pinned (page-locked) CPU memory, which
is what PyTorch's DataLoader does when pin_memory=True. Pinned memory lets the
PCIe DMA engine transfer directly from CPU DRAM without staging through a
bounce buffer, so it gives the highest attainable H2D rate. Using pageable
memory instead would measure a different (slower) path and would not match
what the inference code actually does.

Tensor sizes
------------
SHAPES matches the actual surface, upper-air, and diagnostic tensors in the
S2S inference loop. These are the same sizes recorded by CUPTI in the nsys
profiles and used in d2h_pattern_smoke.py, so the bandwidth numbers here are
directly comparable to the H2D section of compare_nsys.py.

Usage
-----
Run on any GPU node:
    PYTHONPATH=v2.0 python v2.0/test/bandwidth_test.py

To isolate NUMA effects, run twice with numactl and compare:
    numactl --cpunodebind=0 --membind=0 python bandwidth_test.py
    numactl --cpunodebind=1 --membind=1 python bandwidth_test.py

A significant difference between the two runs means the GPU processes are
sensitive to which NUMA node their CPU thread runs on — confirming the NUMA
hypothesis for the DSI cluster's asymmetric bandwidth degradation.
"""

import time
import statistics
import torch
import torch.multiprocessing as mp
from typing import List

# ---------------------------------------------------------------------------
# Tensor shapes
# ---------------------------------------------------------------------------

# These match SURF_SHAPE / UPPER_SHAPE / DIAG_SHAPE in d2h_pattern_smoke.py
# and the actual transfer sizes measured from the nsys CUPTI records.
# Changing these shapes would make the bandwidth numbers incomparable to
# the nsys profile analysis in compare_nsys.py.
SHAPES = {
    "surface":    (1, 16,  128, 256),   # ~12.5 MB — 2D atmospheric fields
    "upper_air":  (1, 104, 128, 256),   # ~81.8 MB — pressure level fields (largest tensor)
    "diagnostic": (1, 4,   128, 256),   #  ~3.1 MB — derived diagnostic variables
}

NREPS  = 50   # number of timed transfers per shape after warmup
WARMUP = 10   # transfers discarded at the start to let CUDA caches settle


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def _mb(shape: tuple) -> float:
    """Return the size in MB of a float32 tensor with the given shape."""
    n = 1
    for d in shape:
        n *= d
    return n * 4 / 1e6


def measure_h2d(device_id: int, shape: tuple, nreps: int, warmup: int) -> List[float]:
    """
    Measure H2D transfer bandwidth for a single GPU.

    Allocates a pinned CPU tensor (src) and an empty GPU tensor (dst) of the
    given shape. Repeatedly copies src → dst using a synchronous blocking copy
    so the wall-clock time between the two synchronize() calls is the true
    transfer time, not just the time to enqueue the DMA command.

    Parameters
    ----------
    device_id : GPU index (0-based)
    shape     : tensor shape tuple
    nreps     : number of measured transfers
    warmup    : number of transfers to discard before timing starts

    Returns
    -------
    List of bandwidth samples in GB/s, one per measured transfer.
    """
    device = torch.device(f"cuda:{device_id}")

    # pin_memory=True allocates in page-locked CPU RAM, enabling direct DMA
    # without a pageable-to-pinned bounce copy in the CUDA driver.
    src = torch.randn(shape, dtype=torch.float32, pin_memory=True)
    dst = torch.empty(shape, dtype=torch.float32, device=device)
    nbytes = src.numel() * 4  # float32 = 4 bytes per element

    results = []
    for i in range(nreps + warmup):
        # Drain the GPU stream before starting the clock so we measure only
        # the copy, not any preceding kernel activity.
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        # non_blocking=False makes this a synchronous copy: the call does not
        # return until the DMA is complete. This matches how inference.py moves
        # data to the GPU (x.to(device) without non_blocking).
        dst.copy_(src, non_blocking=False)

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        if i >= warmup:
            results.append(nbytes / elapsed / 1e9)  # bytes/s → GB/s

    return results


# ---------------------------------------------------------------------------
# Sequential test
# ---------------------------------------------------------------------------

def sequential_test(num_gpus: int) -> dict:
    """
    Test each GPU independently with all others idle.

    This gives the maximum attainable H2D bandwidth per GPU — no PCIe
    contention, no competing DMA engines. Use this as the baseline to
    compare against the concurrent results.

    Returns a nested dict: {tensor_name: {gpu_id: [bandwidth_samples]}}
    """
    results = {}
    for name, shape in SHAPES.items():
        results[name] = {}
        for gpu in range(num_gpus):
            results[name][gpu] = measure_h2d(gpu, shape, NREPS, WARMUP)
    return results


# ---------------------------------------------------------------------------
# Concurrent test
# ---------------------------------------------------------------------------

def _worker(rank: int, shape: tuple, nreps: int, warmup: int,
            result_queue) -> None:
    """
    Entry point for each child process in the concurrent test.

    Each process owns one GPU (rank == GPU index) and independently measures
    H2D bandwidth for the given shape. Results are passed back to the parent
    via a multiprocessing Queue.

    Using separate OS processes (not threads) is important: it mirrors the
    torchrun setup where each GPU worker is a separate Python process, and it
    avoids the GIL preventing true parallelism. Each process allocates its own
    pinned CPU buffer, so the concurrent test stresses all PCIe lanes and NUMA
    interconnects simultaneously.
    """
    samples = measure_h2d(rank, shape, nreps, warmup)
    result_queue.put((rank, samples))


def concurrent_test(num_gpus: int) -> dict:
    """
    Test all GPUs simultaneously, each in its own process.

    This is the contention test. All processes allocate pinned CPU memory and
    run DMA transfers at the same time. If the node has a single PCIe root
    complex shared across all GPUs, total bandwidth is fixed and each GPU
    gets a fraction. If GPUs are on separate root complexes (or connected via
    NVLink), bandwidth should stay near the sequential baseline.

    Asymmetric degradation — where some GPUs slow down significantly and others
    do not — indicates a NUMA topology issue: the slow GPUs are pulling data
    across a CPU socket boundary rather than from their local NUMA node.

    Returns a nested dict: {tensor_name: {gpu_id: [bandwidth_samples]}}
    """
    results = {}
    for name, shape in SHAPES.items():
        q = mp.Queue()

        # Spawn one child process per GPU, all starting at roughly the same
        # time so their DMA transfers overlap.
        procs = [
            mp.Process(target=_worker, args=(gpu, shape, NREPS, WARMUP, q))
            for gpu in range(num_gpus)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        # Collect results from the queue — order is non-deterministic.
        gpu_results = {}
        while not q.empty():
            gpu_id, samples = q.get()
            gpu_results[gpu_id] = samples
        results[name] = gpu_results

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(label: str, results: dict, num_gpus: int) -> None:
    """Print a per-GPU bandwidth table with one row per tensor type."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    header = f"  {'tensor':<12}  {'size_MB':>7}  " + \
             "  ".join(f"GPU{g} GB/s" for g in range(num_gpus))
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, shape in SHAPES.items():
        mb = _mb(shape)
        row = f"  {name:<12}  {mb:>7.1f}  "
        gpu_data = results.get(name, {})
        for g in range(num_gpus):
            samples = gpu_data.get(g, [])
            if samples:
                row += f"  {statistics.median(samples):>8.2f}  "
            else:
                row += f"  {'N/A':>8}  "
        print(row)

    # Aggregate row: median bandwidth across all tensor types per GPU.
    # This is the single number most comparable to compare_nsys.py's bw_GB/s.
    print()
    row = f"  {'AGGREGATE':<12}  {'':>7}  "
    for g in range(num_gpus):
        all_samples = [s for name in SHAPES for s in results.get(name, {}).get(g, [])]
        if all_samples:
            row += f"  {statistics.median(all_samples):>8.2f}  "
    print(row)


def contention_delta(seq: dict, con: dict, num_gpus: int) -> None:
    """
    Print the per-GPU bandwidth change from sequential to concurrent.

    A negative delta means the GPU is slower under load. Uniform small
    negatives suggest mild shared-bus contention. A large negative on only
    some GPUs is the NUMA signature: those GPUs are on the far socket.
    """
    print(f"\n  Contention delta (concurrent − sequential), median across all tensor sizes:")
    for g in range(num_gpus):
        seq_bw = [s for name in SHAPES for s in seq.get(name, {}).get(g, [])]
        con_bw = [s for name in SHAPES for s in con.get(name, {}).get(g, [])]
        if seq_bw and con_bw:
            delta = statistics.median(con_bw) - statistics.median(seq_bw)
            pct   = delta / statistics.median(seq_bw) * 100
            sign  = "+" if delta >= 0 else ""
            print(f"    GPU{g}: {sign}{delta:.2f} GB/s  ({sign}{pct:.1f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("No CUDA device — this test requires a GPU.")
        return

    num_gpus = torch.cuda.device_count()
    props    = [torch.cuda.get_device_properties(i) for i in range(num_gpus)]

    print(f"\nNode GPU summary:")
    for i, p in enumerate(props):
        print(f"  GPU{i}: {p.name}  ({p.total_memory/1e9:.1f} GB)")

    print(f"\nTransfer sizes (float32, pinned CPU → GPU):")
    for name, shape in SHAPES.items():
        print(f"  {name:<12} {_mb(shape):.1f} MB  shape={shape}")
    print(f"\nWarmup={WARMUP} transfers discarded  |  Measured reps={NREPS}")

    # Sequential — establishes the per-GPU bandwidth ceiling with no contention
    print("\nRunning sequential test (one GPU at a time)...")
    seq = sequential_test(num_gpus)
    print_results("Sequential H2D bandwidth — no contention (one GPU at a time)", seq, num_gpus)

    # Concurrent — all GPUs transferring simultaneously in separate processes,
    # mirroring the actual torchrun multi-process inference setup
    print("\nRunning concurrent test (all GPUs simultaneously, separate processes)...")
    con = concurrent_test(num_gpus)
    print_results("Concurrent H2D bandwidth — under load (all GPUs at once)", con, num_gpus)

    # Delta — key diagnostic: symmetric drop = PCIe contention,
    # asymmetric drop = NUMA penalty on specific GPUs
    contention_delta(seq, con, num_gpus)
    print()


if __name__ == "__main__":
    # spawn avoids CUDA context inheritance issues when forking.
    # Each child process initialises its own CUDA context cleanly.
    mp.set_start_method("spawn", force=True)
    main()
