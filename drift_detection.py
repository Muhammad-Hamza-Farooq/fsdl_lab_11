import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

# --- logging setup ---
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename="logs/drift_detection.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# --- sample data: training vs production ---
train = np.random.normal(0, 1, 1000)
production = np.random.normal(2, 1, 1000)

# --- KS drift test ---
stat, p_value = ks_2samp(train, production)

print("KS statistic:", round(stat, 4))
print("P-value:", round(p_value, 6))
logger.info("KS statistic: %.4f", stat)
logger.info("P-value: %.6f", p_value)

# --- interpret p-value ---
ALPHA = 0.05
if p_value < ALPHA:
    message = f"DRIFT DETECTED (p={p_value:.6f} < {ALPHA})"
    print("ALERT:", message)
    logger.warning(message)
else:
    message = f"No significant drift (p={p_value:.6f} >= {ALPHA})"
    print("OK:", message)
    logger.info(message)

# --- visualize distributions ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(train, bins=30, alpha=0.6, label="Training", color="#2563eb")
ax.hist(production, bins=30, alpha=0.6, label="Production", color="#f97316")
ax.set_xlabel("Value")
ax.set_ylabel("Count")
ax.set_title(f"Data Drift Check (p-value={p_value:.4f})")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

print("Log saved to logs/drift_detection.log")
