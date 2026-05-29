import time

import torch
from torch.utils.data import DataLoader, TensorDataset

x = torch.randn(10000, 100)
y = torch.randint(0, 2, (10000,))
dataset = TensorDataset(x, y)


def benchmark(num_workers, pin_memory):
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    start = time.perf_counter()
    for batch in loader:
        pass
    elapsed = time.perf_counter() - start

    batches = len(loader)
    batches_per_sec = batches / elapsed if elapsed > 0 else 0
    return elapsed, batches_per_sec


def main():
    configs = [
        (0, False),
        (2, False),
        (4, False),
        (0, True),
        (2, True),
        (4, True),
    ]

    print("=== DataLoader Benchmark ===\n")
    print(
        f"{'num_workers':<14} {'pin_memory':<12} {'time (s)':<12} {'batches/sec':<14}"
    )
    print("-" * 54)

    results = []

    for num_workers, pin_memory in configs:
        elapsed, batches_per_sec = benchmark(num_workers, pin_memory)
        results.append((num_workers, pin_memory, elapsed, batches_per_sec))
        print(
            f"{num_workers:<14} "
            f"{str(pin_memory):<12} "
            f"{elapsed:<12.4f} "
            f"{batches_per_sec:<14.1f}"
        )

    best = min(results, key=lambda r: r[2])
    print("\n=== Fastest configuration ===")
    print(
        f"num_workers={best[0]}, pin_memory={best[1]} "
        f"-> {best[2]:.4f}s ({best[3]:.1f} batches/sec)"
    )

    # Lab starter config (num_workers=0, pin_memory=False)
    print("\n=== Lab default (num_workers=0) ===")
    loader = DataLoader(dataset, batch_size=32, num_workers=0)
    for batch in loader:
        pass
    print("completed")


if __name__ == "__main__":
    main()
