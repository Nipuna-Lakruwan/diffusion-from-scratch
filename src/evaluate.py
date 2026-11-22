"""Score generated digits with the MNIST judge network.

Metrics
  FD       Frechet distance between judge features of generated vs. real test digits (lower = closer)
  conf     share of samples the judge classifies with >= 90% confidence ("clearly a digit")
  acc      (conditional only) share of samples the judge reads as the digit we asked for
  balance  entropy of the predicted-class histogram / log(10): 1.0 = all digits equally common

    uv run python src/evaluate.py --n 2000
"""
import argparse
import json
import time

import numpy as np
import torch
from scipy import linalg
from torch.utils.data import DataLoader

from classifier import load_judge
from common import CKPT, RESULTS, device, mnist
from diffusion import Diffusion
from unet import UNet


def frechet(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if isinstance(covmean, tuple):  # older SciPy returns (sqrtm, error)
        covmean = covmean[0]
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


@torch.no_grad()
def judge_stats(judge, x, dev, y=None):
    feats, probs = [], []
    for i in range(0, len(x), 500):
        b = x[i:i + 500].to(dev)
        f = judge.features(b)
        feats.append(f.cpu()); probs.append(judge.head(f).softmax(1).cpu())
    feats, probs = torch.cat(feats).numpy(), torch.cat(probs)
    pred = probs.argmax(1)
    hist = torch.bincount(pred, minlength=10).float() / len(pred)
    ent = float(-(hist[hist > 0] * hist[hist > 0].log()).sum() / np.log(10))
    out = {"conf": float((probs.max(1).values >= 0.9).float().mean()), "balance": ent}
    if y is not None:
        out["acc"] = float((pred == y.cpu()).float().mean())
    return feats, out


def load_model(dev):
    s = torch.load(CKPT / "cond_ema.pt", map_location=dev)
    m = UNet(num_classes=10).to(dev)
    m.load_state_dict(s["ema"])
    return m.eval(), s["args"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=500)
    args = ap.parse_args()
    dev = device()
    judge = load_judge(dev)
    model, targs = load_model(dev)
    diff = Diffusion(T=targs["T"], device=dev)

    real = torch.cat([x for x, _ in DataLoader(mnist(False), batch_size=1000)])
    real_feats, real_stats = judge_stats(judge, real, dev)
    train_real = torch.cat([x for x, _ in DataLoader(mnist(True), batch_size=1000)])[: args.n]
    tr_feats, tr_stats = judge_stats(judge, train_real, dev)
    rows = [{"name": f"real train digits (floor, n={args.n})", "FD": frechet(tr_feats, real_feats), **tr_stats}]

    def run(name, fn, y=None):
        g = torch.Generator(device=dev).manual_seed(0)
        t0, xs = time.time(), []
        for i in range(0, args.n, args.bs):
            yb = None if y is None else y[i:i + args.bs]
            xs.append(fn(min(args.bs, args.n - i), yb, g).cpu())
        secs = time.time() - t0
        x = torch.cat(xs)
        f, st = judge_stats(judge, x, dev, y)
        row = {"name": name, "FD": frechet(f, real_feats), **st, "sec_per_1k": secs / args.n * 1000}
        print(row, flush=True)
        rows.append(row)

    y = torch.arange(10, device=dev).repeat_interleave(args.n // 10)
    # unconditional (label hidden): full 400-step DDPM vs 50-step DDIM
    run("ddpm400 unconditional", lambda n, yb, g: diff.sample(model, n, generator=g)[0])
    run("ddim50 unconditional", lambda n, yb, g: diff.sample_ddim(model, n, 50, generator=g))
    for steps in (10, 25, 100):
        run(f"ddim{steps} unconditional", lambda n, yb, g, s=steps: diff.sample_ddim(model, n, s, generator=g))
    # conditional with classifier-free guidance strength w
    for w in (0.0, 1.0, 2.0, 4.0):
        run(f"ddim50 cond w={w}", lambda n, yb, g, w=w: diff.sample_ddim(model, n, 50, y=yb, guidance=w, generator=g), y)
    run("ddpm400 cond w=2.0", lambda n, yb, g: diff.sample(model, n, y=yb, guidance=2.0, generator=g)[0], y)

    (RESULTS / "metrics.json").write_text(json.dumps({"n": args.n, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
