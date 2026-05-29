import logging

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

metrics = {
    "latency": 120,
    "accuracy": 0.91,
    "throughput": 500,
}

THRESHOLDS = {
    "latency": 150,
    "accuracy": 0.85,
    "throughput": 300,
}


def detect_anomalies(values):
    alerts = []
    for name, value in values.items():
        if name == "latency" and value > THRESHOLDS[name]:
            alerts.append(f"High latency: {value}")
        if name == "accuracy" and value < THRESHOLDS[name]:
            alerts.append(f"Low accuracy: {value}")
        if name == "throughput" and value < THRESHOLDS[name]:
            alerts.append(f"Low throughput: {value}")
    return alerts


def simulate_monitoring_failure():
    logger.error("Simulated monitoring failure: metrics collector unreachable")
    return {"latency": None, "accuracy": None, "throughput": None}


def main():
    print("=== Task 21: Monitoring Dashboards ===\n")
    print("Current metrics:", metrics)

    alerts = detect_anomalies(metrics)
    if alerts:
        for alert in alerts:
            logger.warning("ALERT: %s", alert)
    else:
        logger.info("All metrics within thresholds")

    # metric plots
    names = list(metrics.keys())
    values = list(metrics.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, values, color=["#2563eb", "#16a34a", "#f97316"])
    ax.set_title("Production Metrics Dashboard")
    ax.set_ylabel("Value")
    plt.tight_layout()
    plt.show()

    failed = simulate_monitoring_failure()
    print("Monitoring failure simulation:", failed)


if __name__ == "__main__":
    main()
