"""Train the CNN on CIFAR10 and save weights to app/cnn/model.pth.

Run from the repository root:
    uv run --group train python -m training.train
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from app.cnn.model import SimpleCNN
from app.cnn.preprocess import transform
from training.cifar10_data import CIFAR10HF

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "app" / "cnn" / "model.pth"


def main(epochs: int = 5, batch_size: int = 64, lr: float = 1e-3):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    # Official download server is slow; load the same data from a Hugging Face mirror.
    train_set = CIFAR10HF(DATA_DIR, train=True, transform=transform)
    test_set = CIFAR10HF(DATA_DIR, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        print(f"epoch {epoch + 1}/{epochs}  loss={running / len(train_loader):.4f}  "
              f"test_acc={correct / total:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
