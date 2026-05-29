import gc
import os
import time

import psutil
import torch

PROCESS = psutil.Process(os.getpid())


def get_ram_mb():
    return PROCESS.memory_info().rss / (1024 * 1024)


def get_gpu_info():
    if not torch.cuda.is_available():
        return "CUDA not available"

    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    return f"GPU allocated={allocated:.1f} MB, reserved={reserved:.1f} MB"


def stress_batch(batch_size, features=1000):
    before = get_ram_mb()
    start = time.perf_counter()

    x = torch.randn(batch_size, features)
    y = torch.matmul(x, x.T)

    elapsed = time.perf_counter() - start
    after = get_ram_mb()

    print(f"batch_size={batch_size:>6} | shape={tuple(x.shape)} | "
          f"RAM {before:.1f}->{after:.1f} MB | time={elapsed:.3f}s | {get_gpu_info()}")

    del x, y
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return after - before


def detect_memory_leak_runs(runs=3, batch_size=5000):
    print("\n=== Memory leak check (repeat same batch) ===")
    deltas = []
    for i in range(runs):
        delta = stress_batch(batch_size, features=500)
        deltas.append(delta)
        print(f"  run {i + 1} RAM increase: {delta:.1f} MB")

    if deltas[-1] > deltas[0] * 2 and deltas[-1] > 50:
        print("WARNING: Possible memory leak detected")
    else:
        print("OK: No significant memory leak detected")


def main():
    print("=== Task 18: Stress Test Training Pipeline ===\n")
    print(f"Start RAM: {get_ram_mb():.1f} MB")
    print(f"Start GPU: {get_gpu_info()}\n")

    # lab starter stress (large tensor)
    print("=== Lab stress tensor ===")
    x = torch.randn(100000, 1000)
    print(f"Large tensor shape: {tuple(x.shape)}")
    print(f"RAM after large tensor: {get_ram_mb():.1f} MB")
    del x
    gc.collect()

    # optimize batch sizes
    print("\n=== Batch size comparison ===")
    for batch in [1000, 5000, 10000, 20000]:
        try:
            stress_batch(batch, features=1000)
        except RuntimeError as err:
            print(f"batch_size={batch} FAILED: {err}")
            break

    detect_memory_leak_runs()

    print(f"\nFinal RAM: {get_ram_mb():.1f} MB")
    print("Stress test complete")


if __name__ == "__main__":
    main()
