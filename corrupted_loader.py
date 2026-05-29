import logging
import shutil
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT = Path("checkpoints/broken_model.pt")
BACKUP = Path("checkpoints/backup_model.pt")
GOOD = Path("checkpoints/good_model.pt")


def create_fixtures():
    Path("checkpoints").mkdir(exist_ok=True)
    CHECKPOINT.write_text("this is not a valid pytorch file", encoding="utf-8")
    torch.save({"weights": torch.randn(3)}, GOOD)
    shutil.copy(GOOD, BACKUP)
    logger.info("Fixtures created")


def validate_checkpoint(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {path}")
    if path.stat().st_size < 10:
        raise ValueError(f"Checkpoint too small: {path}")
    return True


def load_checkpoint(path):
    validate_checkpoint(path)
    return torch.load(path, weights_only=False)


def load_with_recovery(primary, backup, rollback):
    try:
        logger.info("Loading primary checkpoint: %s", primary)
        return load_checkpoint(primary)
    except Exception as err:
        logger.error("Primary checkpoint failed: %s", err)
        try:
            logger.warning("Trying backup: %s", backup)
            return load_checkpoint(backup)
        except Exception as backup_err:
            logger.error("Backup failed: %s", backup_err)
            logger.warning("Rolling back to last known good: %s", rollback)
            return load_checkpoint(rollback)


def main():
    create_fixtures()

    print("=== Task 19: Handle Corrupted Checkpoints ===\n")
    model = load_with_recovery(CHECKPOINT, BACKUP, GOOD)
    print("Recovery successful. Loaded keys:", list(model.keys()))


if __name__ == "__main__":
    main()
