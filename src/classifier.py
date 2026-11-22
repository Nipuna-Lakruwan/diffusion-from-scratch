"""A small MNIST CNN used only as a *judge* of generated digits.

Its 128-d penultimate features are used for a Frechet distance between generated and real digits
(the same formula as FID, but with an MNIST network instead of Inception, so numbers are not
comparable to published FID on natural images).

    uv run python src/classifier.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import CKPT, mnist


class Judge(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 8 * 8, 128), nn.ReLU())
        self.head = nn.Linear(128, 10)

    def features(self, x):
        return self.conv(x)

    def forward(self, x):
        return self.head(self.features(x))


def load_judge(dev):
    m = Judge().to(dev)
    m.load_state_dict(torch.load(CKPT / "judge.pt", map_location=dev))
    return m.eval()


def main(epochs=4):
    dev = "cpu"  # the GPU is busy training the diffusion model; this takes a few minutes on CPU
    torch.manual_seed(0)
    m = Judge().to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    tr = DataLoader(mnist(True), batch_size=256, shuffle=True)
    for ep in range(epochs):
        m.train()
        for x, y in tr:
            # light augmentation so the judge isn't fooled by tiny shifts
            x = torch.roll(x, shifts=tuple(torch.randint(-2, 3, (2,)).tolist()), dims=(2, 3))
            loss = F.cross_entropy(m(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        correct = 0
        with torch.no_grad():
            for x, y in DataLoader(mnist(False), batch_size=1000):
                correct += (m(x).argmax(1) == y).sum().item()
        print(f"epoch {ep + 1}: test accuracy {correct / 100:.2f}%", flush=True)
    CKPT.mkdir(exist_ok=True)
    torch.save(m.state_dict(), CKPT / "judge.pt")


if __name__ == "__main__":
    main()
