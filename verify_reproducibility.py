import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(seed):
    set_seed(seed)
    output = torch.randn(3)
    print(f"Run output (seed={seed}): {output.tolist()}")
    return output.tolist()


def save_config_copy(config, folder="experiments"):
    Path(folder).mkdir(exist_ok=True)
    path = Path(folder) / "saved_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    print(f"Config saved: {path}")
    return path


def save_run_output(run_id, output, folder="experiments"):
    Path(folder).mkdir(exist_ok=True)
    path = Path(folder) / f"run_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"output": output}, f, indent=2)
    print(f"Run output saved: {path}")
    return path


def compare_outputs(output_a, output_b):
    match = output_a == output_b
    print("\n=== Compare Experiment Outputs ===")
    print("Run 1:", output_a)
    print("Run 2:", output_b)
    if match:
        print("RESULT: Outputs MATCH - experiment is reproducible")
    else:
        print("RESULT: Outputs DIFFER - experiment is NOT reproducible")
    return match


def main():
    config = load_config()
    seed = config["experiment"]["seed"]

    print("=== Task 17: Verify Reproducibility ===\n")
    save_config_copy(config)

    # run multiple times with same seed
    run1 = run_experiment(seed)
    run2 = run_experiment(seed)

    save_run_output(1, run1)
    save_run_output(2, run2)

    compare_outputs(run1, run2)


if __name__ == "__main__":
    main()
