import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

# main production log
logging.basicConfig(
    filename="logs/production.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

# separate ERROR log with rotation
error_handler = RotatingFileHandler(
    "logs/production_errors.log",
    maxBytes=1024,
    backupCount=3,
    encoding="utf-8",
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(error_handler)

logger = logging.getLogger(__name__)


def predict(x, request_id="req-001", endpoint="/predict"):
    logger.info("Request %s on %s with input=%s", request_id, endpoint, x)
    try:
        result = 100 / x
        logger.info("Prediction successful for %s: %s", request_id, result)
        return result
    except Exception:
        logger.exception("Prediction failed for %s with input=%s", request_id, x)
        return None


if __name__ == "__main__":
    predict(10, request_id="req-100")
    predict(0, request_id="req-101")
    print("Logs written to logs/production.log and logs/production_errors.log")
