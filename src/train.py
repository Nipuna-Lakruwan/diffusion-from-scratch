"""Train the noise-predicting U-Net on MNIST.

    uv run python src/train.py --epochs 30            # class-conditional (CFG) model
    uv run python src/train.py --epochs 30 --uncond   # unconditional model
"""
import argparse
import copy
import json
import time

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from common import CKPT, RESULTS, device, mnist, to_img
from diffusion import Diffusion
from unet import UNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--uncond", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = device()
    name = "uncond" if args.uncond else "cond"
    CKPT.mkdir(exist_ok=True)
    snap_dir = RESULTS / f"train_snapshots_{name}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(mnist(True), batch_size=args.bs, shuffle=True, drop_last=True, num_workers=2,
                        persistent_workers=True)
    model = UNet(num_classes=0 if args.uncond else 10).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)
    diff = Diffusion(T=args.T, device=dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  device: {dev}")

    start_epoch, log = 0, []
    ck = CKPT / f"{name}_last.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"])
        opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        start_epoch, log = s["epoch"] + 1, s["log"]
        print(f"resumed from epoch {start_epoch}")

    g = torch.Generator(device=dev)
    fixed_y = torch.arange(10, device=dev).repeat_interleave(8) if not args.uncond else None
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(dev), y.to(dev)
            loss = diff.loss(model, x, None if args.uncond else y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            with torch.no_grad():  # exponential moving average of the weights -> cleaner samples
                decay = min(0.999, (1 + epoch * len(loader) + i) / (10 + epoch * len(loader) + i))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - decay)
            run += loss.item()
            if i % 100 == 0:
                log.append({"step": epoch * len(loader) + i, "loss": loss.item()})
        dt = time.time() - t0
        print(f"epoch {epoch + 1}/{args.epochs} loss {run / len(loader):.4f} ({dt:.0f}s)", flush=True)

        g.manual_seed(1234)  # same starting noise every epoch -> watch the model improve
        imgs = diff.sample_ddim(ema, 80, steps=50, y=fixed_y, guidance=2.0 if fixed_y is not None else 0, generator=g)
        save_image(to_img(imgs), snap_dir / f"epoch_{epoch + 1:03d}.png", nrow=8, padding=1)  # one row per digit

        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "log": log, "args": vars(args)}, ck)
    torch.save({"ema": ema.state_dict(), "args": vars(args)}, CKPT / f"{name}_ema.pt")
    (RESULTS / f"loss_{name}.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
