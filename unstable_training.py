import logging

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_NORM = 1.0
MAX_EPOCHS = 100


def train_with_safeguards():
    x = torch.tensor([1.0], requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=0.1)

    print("=== Training with NaN safeguards ===\n")

    for epoch in range(MAX_EPOCHS):
        optimizer.zero_grad()

        # unstable operation: x^10 causes exploding gradients
        y = x**10
        loss = y.sum()
        loss.backward()

        grad = x.grad.item()
        logger.info("Epoch %s | loss=%.6f | grad=%.6f", epoch + 1, loss.item(), grad)
        print(f"Epoch {epoch + 1} | loss={loss.item():.6f} | grad={grad:.6f}")

        # detect NaN / Inf gradients
        if torch.isnan(x.grad).any() or torch.isinf(x.grad).any():
            msg = f"NaN/Inf gradient detected at epoch {epoch + 1}. Stopping training."
            logger.error(msg)
            print(f"STOP: {msg}")
            break

        # gradient clipping
        torch.nn.utils.clip_grad_norm_([x], MAX_NORM)

        optimizer.step()

        # safe stopping if parameter becomes unstable
        if torch.isnan(x).any() or torch.isinf(x).any() or abs(x.item()) > 1e6:
            msg = f"Unstable parameter value at epoch {epoch + 1}. Stopping training."
            logger.error(msg)
            print(f"STOP: {msg}")
            break
    else:
        print("Training completed without NaN failure.")
        logger.info("Training completed successfully")

    print(f"\nFinal x value: {x.item()}")


if __name__ == "__main__":
    train_with_safeguards()
