"""Write sample metrics for TensorBoard (Task 9)."""
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/mlops_lab")

for epoch in range(10):
    loss = 1.0 / (epoch + 1)
    accuracy = 70 + epoch * 2.5
    writer.add_scalar("Loss/train", loss, epoch)
    writer.add_scalar("Accuracy/train", accuracy, epoch)

writer.close()
print("TensorBoard logs saved to runs/mlops_lab")
