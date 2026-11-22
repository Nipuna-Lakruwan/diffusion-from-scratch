"""Make every figure and GIF in results/ from the trained model.

    uv run python src/figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.utils import make_grid, save_image

from common import RESULTS, device, mnist, to_img
from diffusion import Diffusion
from evaluate import load_model

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})


def grid_np(x, nrow, pad=1):
    g = make_grid(to_img(x), nrow=nrow, padding=pad, pad_value=0.15)
    return (g[0].numpy() * 255).astype(np.uint8)


def forward_noising(diff):
    ds = mnist(False)
    x0 = torch.stack([ds[i][0] for i in (0, 1, 2, 3)])  # 7, 2, 1, 0
    ts = [0, 20, 50, 100, 200, 399]
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(x0.shape, generator=g)
    fig, axes = plt.subplots(len(x0), len(ts), figsize=(len(ts) * 1.3, len(x0) * 1.4))
    for j, t in enumerate(ts):
        tt = torch.full((len(x0),), t)
        xt = diff.q_sample(x0, tt, noise)
        for i in range(len(x0)):
            ax = axes[i, j]
            ax.imshow(to_img(xt[i, 0]), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"t = {t}\n" + r"$\sqrt{\bar\alpha_t}$=" + f"{diff.sqrt_ab[t]:.2f}", fontsize=8)
    fig.suptitle("Forward process: noise is added on purpose (no training needed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "forward_noising.png")
    plt.close(fig)


def sampling_gif(model, diff, dev):
    g = torch.Generator(device=dev).manual_seed(7)
    y = torch.arange(10, device=dev).repeat(4)  # 4 rows of 0..9
    _, frames = diff.sample(model, 40, y=y, guidance=2.0, record_every=5, generator=g)
    imgs, durs = [], []
    for t, x in frames:
        # pure-noise frames barely change and compress badly: show every 4th while t > 100
        if t > 100 and t % 20 != 0 and t != frames[0][0]:
            continue
        arr = grid_np(x, nrow=10, pad=2)
        im = Image.fromarray(arr).resize((arr.shape[1] * 3, arr.shape[0] * 3), Image.NEAREST).convert("L")
        canvas = Image.new("L", (im.width, im.height + 26), 14)
        canvas.paste(im, (0, 26))
        ImageDraw.Draw(canvas).text((8, 6), f"reverse diffusion  t = {t:3d}", fill=230)
        imgs.append(canvas.quantize(32))  # grayscale, 32 levels: keeps the GIF small
        durs.append(120 if t > 100 else 60)
    imgs.append(imgs[-1]); durs.append(1500)  # hold the final frame
    imgs[0].save(RESULTS / "sampling.gif", save_all=True, append_images=imgs[1:], duration=durs, loop=0,
                 optimize=True)
    # static strip for the README
    pick = [0, 20, 40, 60, 70, 76, len(frames) - 1]
    fig, axes = plt.subplots(1, len(pick), figsize=(len(pick) * 1.6, 1.9))
    for ax, k in zip(axes, pick):
        t, x = frames[k]
        ax.imshow(to_img(x[3, 0]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t = {t}", fontsize=9); ax.axis("off")
    fig.suptitle("Pure noise in, a new '3' out", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "reverse_strip.png")
    plt.close(fig)


def sample_grids(model, diff, dev):
    g = torch.Generator(device=dev).manual_seed(1)
    x, _ = diff.sample(model, 100, generator=g)
    save_image(to_img(x), RESULTS / "samples_unconditional.png", nrow=10, padding=1, pad_value=0.15)
    g.manual_seed(2)
    y = torch.arange(10, device=dev).repeat_interleave(10)
    x, _ = diff.sample(model, 100, y=y, guidance=2.0, generator=g)
    save_image(to_img(x), RESULTS / "samples_conditional.png", nrow=10, padding=1, pad_value=0.15)
    # same noise, increasing guidance
    rows = []
    y = torch.arange(10, device=dev)
    for w in (0.0, 1.0, 2.0, 4.0, 8.0):
        g.manual_seed(3)
        rows.append(diff.sample_ddim(model, 10, 50, y=y, guidance=w, generator=g).cpu())
    fig, ax = plt.subplots(figsize=(7, 3.9))
    ax.imshow(grid_np(torch.cat(rows), nrow=10, pad=2), cmap="gray")
    ax.set_yticks([17 + 34 * i for i in range(5)], [f"w = {w}" for w in (0, 1, 2, 4, 8)])
    ax.set_xticks([17 + 34 * i for i in range(10)], [str(i) for i in range(10)])
    ax.set_title("Classifier-free guidance: same starting noise, stronger 'prompt' going down")
    fig.tight_layout()
    fig.savefig(RESULTS / "guidance_sweep.png")
    plt.close(fig)


def training_gif():
    snaps = sorted((RESULTS / "train_snapshots_cond").glob("epoch_*.png"))
    if not snaps:
        return
    imgs = []
    for p in snaps:
        im = Image.open(p).convert("RGB")
        im = im.resize((im.width * 2, im.height * 2), Image.NEAREST)
        canvas = Image.new("RGB", (im.width, im.height + 24), (12, 12, 16))
        canvas.paste(im, (0, 24))
        ImageDraw.Draw(canvas).text((8, 5), f"after epoch {p.stem.split('_')[1].lstrip('0')}", fill=(230, 230, 230))
        imgs.append(canvas)
    imgs += [imgs[-1]] * 4
    imgs[0].save(RESULTS / "training_progress.gif", save_all=True, append_images=imgs[1:], duration=400, loop=0)


def loss_curve():
    log = json.loads((RESULTS / "loss_cond.json").read_text())
    s = np.array([r["step"] for r in log]); l = np.array([r["loss"] for r in log])
    k = 10
    sm = np.convolve(l, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(s, l, alpha=0.3, lw=0.8, label="batch loss")
    ax.plot(s[k - 1:], sm, lw=1.6, label=f"moving avg ({k})")
    ax.set_xlabel("training step"); ax.set_ylabel(r"$\|\epsilon-\epsilon_\theta(x_t,t)\|^2$")
    ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Noise-prediction loss")
    fig.tight_layout(); fig.savefig(RESULTS / "loss_curve.png"); plt.close(fig)


def metrics_plots():
    p = RESULTS / "metrics.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())["rows"]
    steps = [(int(r["name"].split()[0][4:]), r) for r in rows if r["name"].startswith("ddim") and "uncond" in r["name"]]
    steps.sort()
    cond = [(float(r["name"].split("w=")[1]), r) for r in rows if r["name"].startswith("ddim50 cond")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.3))
    a1.plot([s for s, _ in steps], [r["FD"] for _, r in steps], "o-", label="DDIM")
    ddpm = next(r for r in rows if r["name"] == "ddpm400 unconditional")
    a1.axhline(ddpm["FD"], color="C1", ls="--", label="DDPM, 400 steps")
    floor = rows[0]["FD"]
    a1.axhline(floor, color="gray", ls=":", label="real vs real")
    a1.set_xscale("log"); a1.set_xlabel("sampling steps"); a1.set_ylabel("Frechet distance (lower = better)")
    a1.set_title("Fewer steps, same model (unconditional)"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    a2.plot([w for w, _ in cond], [r["acc"] * 100 for _, r in cond], "o-", color="C2", label="judge reads the asked digit (%)")
    a2.set_xlabel("guidance strength w"); a2.set_ylabel("accuracy (%)", color="C2")
    a3 = a2.twinx()
    a3.plot([w for w, _ in cond], [r["FD"] for _, r in cond], "s--", color="C3")
    a3.set_ylabel("Frechet distance", color="C3")
    a2.set_title("Guidance: more on-prompt, less varied"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "metrics.png"); plt.close(fig)


def main():
    dev = device()
    model, targs = load_model(dev)
    diff = Diffusion(T=targs["T"], device=dev)
    forward_noising(Diffusion(T=targs["T"]))
    loss_curve()
    training_gif()
    sampling_gif(model, diff, dev)
    sample_grids(model, diff, dev)
    metrics_plots()


if __name__ == "__main__":
    main()
