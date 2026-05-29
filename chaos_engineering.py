import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def retry(action_name, func, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("%s attempt %s/%s", action_name, attempt, MAX_RETRIES)
            return func(*args, **kwargs)
        except Exception as err:
            logger.warning("%s failed: %s", action_name, err)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(0.2)


def load_dataset(path="data/train.csv"):
    if not Path(path).exists():
        raise FileNotFoundError("Missing dataset")
    return ["row1", "row2"]


def load_config(path="config.yaml"):
    if path == "bad_config.yaml":
        raise ValueError("Wrong config file")
    return {"seed": 42}


def use_gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    return "cuda"


def write_artifact(path="artifacts/model.pt"):
    Path("artifacts").mkdir(exist_ok=True)
    Path(path).write_text("ok", encoding="utf-8")
    return path


def load_artifact(path="artifacts/model.pt"):
    data = Path(path).read_text(encoding="utf-8")
    if data != "ok":
        raise ValueError("Corrupted artifact")
    return data


def network_call(fail_times=2):
    network_call.counter += 1
    if network_call.counter <= fail_times:
        raise TimeoutError("Network timeout")
    return "success"


network_call.counter = 0


def run_resilient_pipeline():
    print("=== Task 20: Chaos Engineering ===\n")

    # missing dataset -> fallback path
    try:
        data = retry("load_dataset", load_dataset)
    except Exception:
        logger.warning("Using fallback in-memory dataset")
        data = ["fallback"]

    # wrong config -> default config
    try:
        cfg = retry("load_config", load_config, "bad_config.yaml")
    except Exception:
        logger.warning("Using default config")
        cfg = {"seed": 42}

    # GPU unavailable -> CPU fallback
    try:
        device = retry("gpu_check", use_gpu)
    except Exception:
        logger.warning("Falling back to CPU")
        device = "cpu"

    # corrupted artifact -> rewrite then load
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/model.pt").write_text("broken", encoding="utf-8")
    try:
        retry("load_artifact", load_artifact)
    except Exception:
        logger.warning("Recovering corrupted artifact")
        write_artifact()

    result = retry("network_call", network_call)
    logger.info("Pipeline completed on %s with %s records (%s)", device, len(data), result)
    print("Pipeline completed with graceful recovery")


if __name__ == "__main__":
    run_resilient_pipeline()
