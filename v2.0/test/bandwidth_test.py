#!/usr/bin/env python3
"""
PCIe host-to-device bandwidth test for multi-GPU nodes.

Measures H2D transfer bandwidth per GPU at the tensor sizes actually used
during S2S inference, under two conditions:
  1. Sequential — each GPU tested independently (no contention)
  2. Concurrent — all GPUs transferring simultaneously (contention under load)

Run on any node with multiple GPUs:
    python bandwidth_test.py

To test NUMA binding effects, run twice with numactl:
    numactl --cpunodebind=0 --membind=0 python bandwidth_test.py
    numactl --cpunodebind=1 --membind=1 python bandwidth_test.py

Output is a table directly comparable to the H2D section of compare_nsys.py.
"""

import time
import statistics
import torch
import torch.multiprocessing as mp
from typing import List, Tuple

# Tensor shapes from a standard S2S inference run at default resolution.
# These match the SURF_SHAPE / UPPER_SHAPE / DIAG_SHAPE in d2h_pattern_smoke.py
# and the actual CUPTI memcpy sizes observed in the nsys profiles.
SHAPES = {
    "surface":    (1, 16,  128, 256),   # ~12.5 MB per transfer
    "upper_air":  (1, 104, 128, 256),   # ~81.8 MB per transfer
    "diagnostic": (1, 4,   128, 256),   #  ~3.1 MB per transfer
}

NREPS   = 50   # transfers per measurement
WARMUP  = 10


def _mb(shape: tuple) -> float:
    n = 1
    for d in shape: n *= d
    return n * 4 / 1e6  # float32


def measure_h2d(device_id: int, shape: tuple, nreps: int, warmup: int) -> List[float]:
    """
    Returns per-transfer bandwidth samples (GB/s) for one GPU.
    Uses pinned memory source to match the inference DataLoader behaviour.
    """
    device = torch.device(f"cuda:{device_id}")
    src = torch.randn(shape, dtype=torch.float32, pin_memory=True)
    dst = torch.empty(shape, dtype=torch.float32, device=device)
    nbytes = src.numel() * 4

    results = []
    for i in range(nreps + warmup):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        dst.copy_(src, non_blocking=False)   # synchronous, matches inference default
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        if i >= warmup:
            results.append(nbytes / elapsed / 1e9)

    return results


def _worker(rank: int, shape: tuple, nreps: int, warmup: int,
            result_queue) -> None:
    """Worker for concurrent multi-GPU test — one process per GPU."""
    samples = measure_h2d(rank, shape, nreps, warmup)
    result_queue.put((rank, samples))


def sequential_test(num_gpus: int) -> dict:
    """Test each GPU one at a time — no inter-GPU contention."""
    results = {}
    for name, shape in SHAPES.items():
        results[name] = {}
        for gpu in range(num_gpus):
            samples = measure_h2d(gpu, shape, NREPS, WARMUP)
            results[name][gpu] = samples
    return results


def concurrent_test(num_gpus: int) -> dict:
    """
    All GPUs transfer simultaneously — reveals PCIe contention and NUMA effects.
    Each GPU runs in a separate process to mirror the torchrun multi-process setup.
    """
    results = {}
    for name, shape in SHAPES.items():
        q = mp.Queue()
        procs = [
            mp.Process(target=_worker, args=(gpu, shape, NREPS, WARMUP, q))
            for gpu in range(num_gpus)
        ]
        for p in procs: p.start()
        for p in procs: p.join()

        gpu_results = {}
        while not q.empty():
            gpu_id, samples = q.get()
            gpu_results[gpu_id] = samples
        results[name] = gpu_results
    return results


def print_results(label: str, results: dict, num_gpus: int) -> None:
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

    # Summary: median across all tensors per GPU
    print()
    row = f"  {'AGGREGATE':<12}  {'':>7}  "
    for g in range(num_gpus):
        all_samples = [s for name in SHAPES for s in results.get(name, {}).get(g, [])]
        if all_samples:
            row += f"  {statistics.median(all_samples):>8.2f}  "
    print(row)


def contention_delta(seq: dict, con: dict, num_gpus: int) -> None:
    """Print the bandwidth drop from sequential to concurrent per GPU."""
    print(f"\n  Contention delta (concurrent - sequential), median GB/s:")
    for g in range(num_gpus):
        seq_bw, con_bw = [], []
        for name in SHAPES:
            seq_bw += seq.get(name, {}).get(g, [])
            con_bw += con.get(name, {}).get(g, [])
        if seq_bw and con_bw:
            delta = statistics.median(con_bw) - statistics.median(seq_bw)
            pct   = delta / statistics.median(seq_bw) * 100
            sign  = "+" if delta >= 0 else ""
            print(f"    GPU{g}: {sign}{delta:.2f} GB/s  ({sign}{pct:.1f}%)")


def main():
    if not torch.cuda.is_available():
        print("No CUDA device — this test requires a GPU.")
        return

    num_gpus = torch.cuda.device_count()
    props    = [torch.cuda.get_device_properties(i) for i in range(num_gpus)]

    print(f"\nNode: {torch.cuda.get_device_name(0)}")
    for i, p in enumerate(props):
        print(f"  GPU{i}: {p.name}  ({p.total_memory/1e9:.1f} GB)")

    print(f"\nTransfer sizes:")
    for name, shape in SHAPES.items():
        print(f"  {name:<12} {_mb(shape):.1f} MB  shape={shape}")
    print(f"\nWarmup={WARMUP}  Measured reps={NREPS}")

    print("\nRunning sequential test (one GPU at a time)...")
    seq = sequential_test(num_gpus)
    print_results("Sequential H2D bandwidth (pinned → GPU, one GPU at a time)", seq, num_gpus)

    print("\nRunning concurrent test (all GPUs simultaneously)...")
    con = concurrent_test(num_gpus)
    print_results("Concurrent H2D bandwidth (pinned → GPU, all GPUs at once)", con, num_gpus)

    contention_delta(seq, con, num_gpus)
    print()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
