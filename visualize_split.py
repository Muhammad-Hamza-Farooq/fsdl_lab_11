import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
feature_name = "Petal length (cm)"
feature_values = X[:, 2]

# Broken: 95% test → only 7 training samples
X_train_bad, X_test_bad, _, _ = train_test_split(X, y, test_size=0.95, random_state=42)

# Fixed: 20% test → 120 training samples
X_train_good, X_test_good, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

bins = np.linspace(feature_values.min(), feature_values.max(), 12)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Left: broken split (dual y-axis so tiny train set is visible) ---
ax_train = axes[0]
ax_test = ax_train.twinx()

ax_train.hist(
    X_train_bad[:, 2],
    bins=bins,
    alpha=0.85,
    label=f"Train ({len(X_train_bad)} samples)",
    color="#2563eb",
    edgecolor="navy",
    linewidth=1.2,
    zorder=3,
)
ax_test.hist(
    X_test_bad[:, 2],
    bins=bins,
    alpha=0.55,
    label=f"Test ({len(X_test_bad)} samples)",
    color="#f97316",
    edgecolor="white",
    zorder=2,
)

ax_train.set_ylabel("Train count", color="#2563eb", fontweight="bold")
ax_test.set_ylabel("Test count", color="#f97316", fontweight="bold")
ax_train.tick_params(axis="y", labelcolor="#2563eb")
ax_test.tick_params(axis="y", labelcolor="#f97316")
ax_train.set_ylim(0, max(8, len(X_train_bad) + 1))

ax_train.set_title("Broken split (test_size=0.95)")
ax_train.set_xlabel(feature_name)
ax_train.grid(axis="y", alpha=0.3)

lines1, labels1 = ax_train.get_legend_handles_labels()
lines2, labels2 = ax_test.get_legend_handles_labels()
ax_train.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

# --- Right: fixed split (single axis — both sets visible) ---
axes[1].hist(
    X_train_good[:, 2],
    bins=bins,
    alpha=0.7,
    label=f"Train ({len(X_train_good)} samples)",
    color="#2563eb",
    edgecolor="white",
)
axes[1].hist(
    X_test_good[:, 2],
    bins=bins,
    alpha=0.7,
    label=f"Test ({len(X_test_good)} samples)",
    color="#f97316",
    edgecolor="white",
)
axes[1].set_title("Fixed split (test_size=0.2)")
axes[1].set_xlabel(feature_name)
axes[1].set_ylabel("Count")
axes[1].legend(loc="upper right")
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle(
    "Train vs test distribution (Iris petal length)\n"
    "Broken plot: blue = train (left axis), orange = test (right axis)",
    fontsize=12,
)
plt.tight_layout()
plt.show()
