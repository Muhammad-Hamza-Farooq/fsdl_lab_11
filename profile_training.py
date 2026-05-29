import time

import torch
from torch.profiler import profile

x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)

# CPU profiling
start = time.perf_counter()
with profile() as prof:
    for _ in range(100):
        z = torch.matmul(x, y)
cpu_time = time.perf_counter() - start

print("=== CPU Profiler Results ===")
print(prof.key_averages().table(sort_by="cpu_time_total"))
print(f"\nTotal wall time (CPU): {cpu_time:.4f} seconds")

# GPU profiling (if available)
if torch.cuda.is_available():
    device = torch.device("cuda")
    x_gpu = x.to(device)
    y_gpu = y.to(device)

    start = time.perf_counter()
    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof_gpu:
        for _ in range(100):
            z = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()
    gpu_time = time.perf_counter() - start

    print("\n=== GPU Profiler Results ===")
    print(prof_gpu.key_averages().table(sort_by="cuda_time_total"))
    print(f"\nTotal wall time (GPU): {gpu_time:.4f} seconds")
    print(f"Speedup vs CPU: {cpu_time / gpu_time:.2f}x")
else:
    print("\nCUDA not available on this machine — CPU profiling only.")
