import logging
import random
import time

LOG_FILE = "logs/training.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console)

learning_rate = 0.001

for epoch in range(5):
    loss = round(random.uniform(0.1, 1.0), 4)
    accuracy = round(random.uniform(70, 95), 2)
    val_accuracy = round(random.uniform(68, 94), 2)
    gpu_usage = round(random.uniform(40, 95), 1)

    logging.info(
        "Epoch %s | LR: %s | Loss: %s | Accuracy: %s%% | Val Accuracy: %s%% | GPU: %s%%",
        epoch + 1,
        learning_rate,
        loss,
        accuracy,
        val_accuracy,
        gpu_usage,
    )

    if accuracy < 75:
        logging.warning("Low training accuracy at epoch %s: %s%%", epoch + 1, accuracy)

    time.sleep(0.2)

print(f"Logs saved to {LOG_FILE}")
