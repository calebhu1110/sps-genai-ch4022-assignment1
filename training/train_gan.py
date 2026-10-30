"""Train the GAN on MNIST and save the Generator weights to app/gan/generator.pth.

Run from the repository root:
    uv run python -m training.train_gan                # full training (20 epochs)
    uv run python -m training.train_gan --epochs 5     # quicker
    uv run python -m training.train_gan --fake --epochs 1 --max-batches 20   # smoke test, no download

MNIST is downloaded into data/ (git-ignored) by torchvision.
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.gan.model import Z_DIM, Discriminator, Generator

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GENERATOR_PATH = ROOT / "app" / "gan" / "generator.pth"
SAMPLES_PATH = ROOT / "training" / "gan_samples.png"

# MNIST pixels are in [0, 1]. The Generator ends with Tanh (outputs in [-1, 1]),
# so real images are rescaled to [-1, 1] as well.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def main(epochs: int = 20, batch_size: int = 128, lr: float = 2e-4,
         fake_data: bool = False, max_batches: int | None = None,
         out_path: Path = GENERATOR_PATH):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    if fake_data:  # random images, only to check that the code runs
        dataset = datasets.FakeData(size=2048, image_size=(1, 28, 28),
                                    num_classes=10, transform=transform)
    else:
        dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    G = Generator(Z_DIM).to(device)
    D = Discriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()  # sigmoid + binary cross-entropy in one step
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, Z_DIM, device=device)  # same noise every epoch -> comparable samples

    for epoch in range(epochs):
        G.train()
        d_total = g_total = 0.0
        n = 0
        for i, (real, _) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            real = real.to(device)
            b = real.size(0)
            ones = torch.ones(b, 1, device=device)    # label for "real"
            zeros = torch.zeros(b, 1, device=device)  # label for "fake"

            # ---- 1) Train the Discriminator: real -> 1, fake -> 0 ----
            z = torch.randn(b, Z_DIM, device=device)
            fake = G(z)
            loss_D = (criterion(D(real), ones) + criterion(D(fake.detach()), zeros)) / 2
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ---- 2) Train the Generator: make D say "real" (1) for fakes ----
            loss_G = criterion(D(fake), ones)
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            d_total += loss_D.item()
            g_total += loss_G.item()
            n += 1

        print(f"epoch {epoch + 1}/{epochs}  D loss={d_total / n:.4f}  G loss={g_total / n:.4f}")

        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise).cpu()
        save_image(samples, SAMPLES_PATH, nrow=8, normalize=True, value_range=(-1, 1))

    G.cpu().eval()
    torch.save(G.state_dict(), out_path)  # only the Generator is needed for the API
    print(f"saved {out_path}")
    print(f"sample grid: {SAMPLES_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--fake", action="store_true", help="use random data (smoke test)")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--out", type=Path, default=GENERATOR_PATH)
    a = p.parse_args()
    main(a.epochs, a.batch_size, a.lr, a.fake, a.max_batches, a.out)
