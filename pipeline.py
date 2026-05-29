import logging
import time

logger = logging.getLogger(__name__)


def load_data(source="memory"):
    logger.info("Loading data from %s", source)
    if source == "database":
        # mocked DB payload in tests
        return [1, 2, 3]
    return [1, 2, 3]


def preprocess(data):
    logger.info("Preprocessing %s records", len(data))
    return [x * 2 for x in data]


def predict(data):
    logger.info("Running prediction")
    return sum(data)


def run_pipeline(source="memory"):
    start = time.perf_counter()

    data = load_data(source=source)
    processed = preprocess(data)
    result = predict(processed)

    elapsed = time.perf_counter() - start
    logger.info("Pipeline finished in %.4f seconds", elapsed)

    return {
        "result": result,
        "elapsed_seconds": elapsed,
        "records": len(data),
    }
