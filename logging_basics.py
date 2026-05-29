import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.debug("Debug: loading dataset")
logging.info("Training started")
logging.warning("GPU usage is high")
logging.error("Dataset file missing")
