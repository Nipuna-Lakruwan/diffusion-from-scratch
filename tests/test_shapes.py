import torch

from diffusion import Diffusion
from unet import UNet


def test_unet_shape():
    m = UNet()
    x = torch.randn(2, 1, 32, 32)
    out = m(x, torch.tensor([0, 399]), torch.tensor([3, 10]))
    assert out.shape == x.shape


def test_forward_process_endpoints():
    d = Diffusion(T=400)
    # almost nothing of the image survives the last step
    assert d.alpha_bar[-1] < 1e-3
    assert d.alpha_bar[0] > 0.999
    x0 = torch.zeros(4, 1, 32, 32)
    xt = d.q_sample(x0, torch.full((4,), 399))
    assert abs(xt.std().item() - 1) < 0.1


def test_loss_and_sampling_run():
    torch.manual_seed(0)
    m = UNet(ch=16, mults=(1, 2), attn_res=())
    d = Diffusion(T=10)
    loss = d.loss(m, torch.randn(4, 1, 32, 32), torch.tensor([1, 2, 3, 4]))
    assert loss.item() > 0
    x, frames = d.sample(m, 2, y=torch.tensor([1, 2]), guidance=1.0, record_every=5)
    assert x.shape == (2, 1, 32, 32) and len(frames) >= 2
    assert d.sample_ddim(m, 2, steps=5).shape == (2, 1, 32, 32)
