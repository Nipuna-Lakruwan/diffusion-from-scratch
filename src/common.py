import torch
from torchvision import datasets, transforms

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def mnist(train=True):
    """MNIST padded to 32x32 and scaled to [-1, 1]."""
    tf = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    return datasets.MNIST(DATA, train=train, download=True, transform=tf)


def to_img(x):
    """[-1, 1] tensor -> [0, 1]."""
    return (x.clamp(-1, 1) + 1) / 2
