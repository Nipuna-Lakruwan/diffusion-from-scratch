"""DDPM forward (noising) and reverse (denoising) processes, plus DDIM for fast sampling.

Notation follows Ho et al. 2020, "Denoising Diffusion Probabilistic Models":
    beta_t        noise added at step t
    alpha_t       = 1 - beta_t
    alpha_bar_t   = prod_{s<=t} alpha_s   (how much of the original image survives at step t)
"""
import torch


class Diffusion:
    def __init__(self, T=400, beta_start=1e-4, beta_end=None, device="cpu"):
        self.T = T
        self.device = device
        # Ho et al. use 0.02 for T=1000. With fewer steps each step must add more noise, so scale it:
        # 0.02 * 1000 / 400 = 0.05, which leaves alpha_bar_T ~ 4e-5 (essentially pure noise).
        beta_end = beta_end if beta_end is not None else 0.02 * 1000 / T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_ab = self.alpha_bar.sqrt()
        self.sqrt_1m_ab = (1 - self.alpha_bar).sqrt()

    # ---- forward process: needs no training ---------------------------------
    def q_sample(self, x0, t, noise=None):
        """Jump straight to step t:  x_t = sqrt(ab_t) x0 + sqrt(1 - ab_t) eps."""
        if noise is None:
            noise = torch.randn_like(x0)
        return self.sqrt_ab[t][:, None, None, None] * x0 + self.sqrt_1m_ab[t][:, None, None, None] * noise

    def loss(self, model, x0, y=None, p_uncond=0.1):
        """L = || eps - eps_theta(x_t, t) ||^2 at a random step t."""
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        if y is not None and p_uncond > 0:
            drop = torch.rand(b, device=x0.device) < p_uncond  # sometimes hide the label -> CFG
            y = torch.where(drop, torch.full_like(y, model.num_classes), y)
        return torch.mean((model(xt, t, y) - noise) ** 2)

    # ---- reverse process -----------------------------------------------------
    def _eps(self, model, x, t_batch, y, guidance):
        if y is None or guidance == 0:
            return model(x, t_batch, y)
        null = torch.full_like(y, model.num_classes)
        eps_c, eps_u = model(torch.cat([x, x]), torch.cat([t_batch, t_batch]), torch.cat([y, null])).chunk(2)
        return eps_u + (1 + guidance) * (eps_c - eps_u)

    @torch.no_grad()
    def sample(self, model, n, y=None, guidance=0.0, record_every=None, generator=None):
        """Ancestral DDPM sampling, T steps:
            x_{t-1} = 1/sqrt(a_t) * (x_t - b_t / sqrt(1 - ab_t) * eps_theta) + sigma_t z
        Returns final images and (optionally) snapshots for a GIF."""
        x = torch.randn(n, 1, 32, 32, device=self.device, generator=generator)
        frames = []
        for t in reversed(range(self.T)):
            tb = torch.full((n,), t, device=self.device, dtype=torch.long)
            eps = self._eps(model, x, tb, y, guidance)
            mean = (x - self.betas[t] / self.sqrt_1m_ab[t] * eps) / self.alphas[t].sqrt()
            if t > 0:
                z = torch.randn(x.shape, device=self.device, generator=generator)
                x = mean + self.betas[t].sqrt() * z
            else:
                x = mean
            if record_every and (t % record_every == 0 or t == self.T - 1):
                frames.append((t, x.clamp(-1, 1).cpu()))
        return x.clamp(-1, 1), frames

    @torch.no_grad()
    def sample_ddim(self, model, n, steps=50, y=None, guidance=0.0, generator=None):
        """Deterministic DDIM (Song et al. 2021): same trained model, far fewer steps."""
        ts = torch.linspace(self.T - 1, 0, steps, device=self.device).long()
        x = torch.randn(n, 1, 32, 32, device=self.device, generator=generator)
        for i, t in enumerate(ts):
            tb = torch.full((n,), int(t), device=self.device, dtype=torch.long)
            eps = self._eps(model, x, tb, y, guidance)
            ab = self.alpha_bar[t]
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            # Keep eps consistent with the clipped x0. Without this, early steps (alpha_bar ~ 1e-4) clip x0
            # hard but still move along the unclipped eps, and guided samples fall apart.
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            if i == len(ts) - 1:
                x = x0
            else:
                ab_next = self.alpha_bar[ts[i + 1]]
                x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
        return x.clamp(-1, 1)
