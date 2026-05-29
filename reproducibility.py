import logging
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


def setup_logging(log_path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    return logging.getLogger(__name__)


def main():
    config = load_config()
    seed = config["experiment"]["seed"]
    exp_name = config["experiment"]["name"]
    checkpoint_path = config["paths"]["checkpoint"]
    log_path = config["paths"]["log"]

    set_seed(seed)
    logger = setup_logging(log_path)

    # Same random tensor every run when seed is fixed
    a = torch.randn(3)
    print("Experiment:", exp_name)
    print("Seed:", seed)
    print("Output tensor:", a)

    # Simple "model" checkpoint (weights from seeded tensor)
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": a, "seed": seed, "experiment": exp_name}, checkpoint_path)

    logger.info("Experiment: %s", exp_name)
    logger.info("Seed: %s", seed)
    logger.info("Output tensor: %s", a.tolist())
    logger.info("Checkpoint saved to %s", checkpoint_path)

    print(f"\nCheckpoint saved: {checkpoint_path}")
    print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
