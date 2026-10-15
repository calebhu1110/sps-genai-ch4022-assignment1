"""CIFAR10 loaded from the Hugging Face mirror (uoft-cs/cifar10).

The official download server (cs.toronto.edu) can be extremely slow, so this
module fetches the same dataset as parquet files from Hugging Face instead.
Labels use the same order as torchvision.datasets.CIFAR10.
"""
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

HF_URL = ("https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/"
          "plain_text/{split}-00000-of-00001.parquet")


def _load_arrays(root: Path, split: str):
    cache = root / f"{split}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["images"], d["labels"]

    parquet = root / f"{split}.parquet"
    if not parquet.exists():
        root.mkdir(parents=True, exist_ok=True)
        print(f"downloading {split} split from Hugging Face ...")
        urllib.request.urlretrieve(HF_URL.format(split=split), parquet)

    df = pd.read_parquet(parquet)
    images = np.stack([np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB"))
                       for b in df["img"]])
    labels = df["label"].to_numpy(dtype=np.int64)
    np.savez(cache, images=images, labels=labels)
    return images, labels


class CIFAR10HF(Dataset):
    """Drop-in replacement for datasets.CIFAR10(root, train, transform=...)."""

    def __init__(self, root: str, train: bool = True, transform=None):
        split = "train" if train else "test"
        self.images, self.labels = _load_arrays(Path(root) / "cifar10_hf", split)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])
