import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Input: 3 x 64 x 64
    Conv(16, 3x3, s1, p1) -> ReLU -> MaxPool(2, 2)   => 16 x 32 x 32
    Conv(32, 3x3, s1, p1) -> ReLU -> MaxPool(2, 2)   => 32 x 16 x 16
    Flatten                                          => 8192
    FC(100) -> ReLU -> FC(10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
