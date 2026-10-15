"""Preprocessing shared by training and the API.

Keeping it in ONE place guarantees the model sees identical inputs at
training time and at inference time.
"""
from torchvision import transforms

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
