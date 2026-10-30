"""GAN architecture for MNIST (Assignment 3).

Generator:      noise (B, 100)      -> image (B, 1, 28, 28), values in [-1, 1]
Discriminator:  image (B, 1, 28, 28) -> one real/fake score (B, 1)
"""
import torch
import torch.nn as nn

Z_DIM = 100


class Generator(nn.Module):
    """
    Input: noise vector (B, 100)
    Linear(100 -> 7*7*128), reshape                       => 128 x 7 x 7
    ConvTranspose2d(128 -> 64, k4, s2, p1) -> BN -> ReLU   => 64 x 14 x 14
    ConvTranspose2d(64 -> 1, k4, s2, p1) -> Tanh           => 1 x 28 x 28
    """

    def __init__(self, z_dim: int = Z_DIM):
        super().__init__()
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)                 # (B, 6272)
        x = x.view(-1, 128, 7, 7)      # (B, 128, 7, 7)
        x = self.deconv1(x)            # (B, 64, 14, 14)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)            # (B, 1, 28, 28)
        x = self.tanh(x)
        return x


class Discriminator(nn.Module):
    """
    Input: image (B, 1, 28, 28)
    Conv2d(1 -> 64, k4, s2, p1) -> LeakyReLU(0.2)                 => 64 x 14 x 14
    Conv2d(64 -> 128, k4, s2, p1) -> BN -> LeakyReLU(0.2)         => 128 x 7 x 7
    Flatten -> Linear(6272 -> 1)                                  => 1

    The last layer returns a raw score (logit). Apply sigmoid to get a
    real/fake probability; training uses BCEWithLogitsLoss, which does that
    internally and is numerically more stable.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7 * 7 * 128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)              # (B, 64, 14, 14)
        x = self.act1(x)
        x = self.conv2(x)              # (B, 128, 7, 7)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.flatten(x)            # (B, 6272)
        return self.fc(x)              # (B, 1)


if __name__ == "__main__":
    # Shape check: run with  python -m app.gan.model
    g, d = Generator(), Discriminator()
    z = torch.randn(4, Z_DIM)
    fake = g(z)
    print("Generator out:    ", tuple(fake.shape))        # (4, 1, 28, 28)
    print("Discriminator out:", tuple(d(fake).shape))     # (4, 1)
