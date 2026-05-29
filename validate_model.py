import logging
import sys
import time
import tracemalloc
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- model metrics (from validation run) ---
metrics = {
    "accuracy": 0.90,
    "f1": 0.88,
    "roc_auc": 0.85,
}

# --- thresholds ---
THRESHOLDS = {
    "accuracy": 0.85,
    "f1": 0.80,
    "roc_auc": 0.80,
    "latency_ms": 200,
    "memory_mb": 512,
}


def check_metric(name, value, threshold, higher_is_better=True):
    if higher_is_better:
        passed = value >= threshold
    else:
        passed = value <= threshold

    status = "PASS" if passed else "FAIL"
    logger.info("%s: %.4f (threshold=%.4f) -> %s", name, value, threshold, status)
    print(f"{name}: {value:.4f} (threshold={threshold}) -> {status}")
    return passed


def simulate_inference():
    """Simulate model inference and measure latency + memory."""
    tracemalloc.start()
    start = time.perf_counter()

    # fake inference work
    data = [i * 0.01 for i in range(100_000)]
    result = sum(data)

    latency_ms = (time.perf_counter() - start) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = peak / (1024 * 1024)

    return result, latency_ms, memory_mb


def validate_model():
    print("=== Model Validation Pipeline ===\n")
    results = []

    # accuracy & f1 (lab starter checks)
    results.append(check_metric("accuracy", metrics["accuracy"], THRESHOLDS["accuracy"]))
    results.append(check_metric("f1", metrics["f1"], THRESHOLDS["f1"]))

    # ROC-AUC check
    results.append(check_metric("roc_auc", metrics["roc_auc"], THRESHOLDS["roc_auc"]))

    # latency check
    _, latency_ms, memory_mb = simulate_inference()
    results.append(check_metric("latency_ms", latency_ms, THRESHOLDS["latency_ms"], higher_is_better=False))

    # memory validation
    results.append(check_metric("memory_mb", memory_mb, THRESHOLDS["memory_mb"], higher_is_better=False))

    # deployment approval logic
    print("\n=== Deployment Decision ===")
    if all(results):
        print("APPROVED: Model passed all validation checks.")
        logger.info("Deployment APPROVED")
        return True

    print("REJECTED: Model failed one or more validation checks.")
    logger.warning("Deployment REJECTED")
    return False


if __name__ == "__main__":
    approved = validate_model()
    sys.exit(0 if approved else 1)
