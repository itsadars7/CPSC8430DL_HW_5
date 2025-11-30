#!/usr/bin/env python3
"""
Adarsha Neupane
CPSC 8430 - Deep Learning
HW-5

Implementation of:
    - DDPM 

Usage:
    python train_ddpm.py --epochs 200 --batch 128 --T 1000 --device cuda
"""

import os
import argparse
import random
from itertools import chain
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, utils
import numpy as np

# seed for reproduction
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Time embedding helpers (sinusoidal + MLP)
def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    # (half,)
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )  # [half]
    # (B, 1) * (1, half) -> (B, half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb  # (B, dim)


class TimeMLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
        )

    def forward(self, t_emb):
        return self.net(t_emb)


# UNet-like epsilon model
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t_emb=None):
        """
        x: (B, C, H, W)
        t_emb: (B, time_emb_dim) or None
        """
        h = self.conv1(x)
        h = self.norm1(h)
        if t_emb is not None:
            # (B, out_ch) -> (B, out_ch, 1, 1) and broadcast
            temb = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + temb
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h


class UNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=128, time_emb_dim=256):
        super().__init__()

        self.time_mlp = TimeMLP(time_emb_dim, time_emb_dim)

        # Encoder
        self.conv_in1 = ConvBlock(img_channels, base_channels, time_emb_dim=time_emb_dim)
        self.conv_in2 = ConvBlock(base_channels, base_channels, time_emb_dim=time_emb_dim)
        
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU()
        )  # 32x32 -> 16x16
        self.conv_down1a = ConvBlock(base_channels * 2, base_channels * 2, time_emb_dim=time_emb_dim)
        self.conv_down1b = ConvBlock(base_channels * 2, base_channels * 2, time_emb_dim=time_emb_dim)
        
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU()
        )  # 16x16 -> 8x8
        self.conv_down2a = ConvBlock(base_channels * 4, base_channels * 4, time_emb_dim=time_emb_dim)
        self.conv_down2b = ConvBlock(base_channels * 4, base_channels * 4, time_emb_dim=time_emb_dim)
        
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 8),
            nn.SiLU()
        )  # 8x8 -> 4x4
        self.conv_down3a = ConvBlock(base_channels * 8, base_channels * 8, time_emb_dim=time_emb_dim)
        self.conv_down3b = ConvBlock(base_channels * 8, base_channels * 8, time_emb_dim=time_emb_dim)
        
        self.down4 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 8),
            nn.SiLU()
        )  # 4x4 -> 2x2
        self.conv_down4a = ConvBlock(base_channels * 8, base_channels * 8, time_emb_dim=time_emb_dim)
        self.conv_down4b = ConvBlock(base_channels * 8, base_channels * 8, time_emb_dim=time_emb_dim)
        
        # Bottleneck
        self.conv_mid1 = ConvBlock(base_channels * 8, base_channels * 8, time_emb_dim=time_emb_dim)
        self.conv_mid2 = ConvBlock(base_channels * 8, base_channels * 8, time_emb_dim=time_emb_dim)
        
        # Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 8),
            nn.SiLU()
        )  # 2x2 -> 4x4
        self.conv_up1a = ConvBlock(base_channels * 16, base_channels * 8, time_emb_dim=time_emb_dim)
        self.conv_up1b = ConvBlock(base_channels * 8, base_channels * 8, time_emb_dim=time_emb_dim)
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU()
        )  # 4x4 -> 8x8
        self.conv_up2a = ConvBlock(base_channels * 8, base_channels * 4, time_emb_dim=time_emb_dim)
        self.conv_up2b = ConvBlock(base_channels * 4, base_channels * 4, time_emb_dim=time_emb_dim)
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU()
        )  # 8x8 -> 16x16
        self.conv_up3a = ConvBlock(base_channels * 4, base_channels * 2, time_emb_dim=time_emb_dim)
        self.conv_up3b = ConvBlock(base_channels * 2, base_channels * 2, time_emb_dim=time_emb_dim)
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )  # 16*16 -> 32x32
        self.conv_up4a = ConvBlock(base_channels * 2, base_channels, time_emb_dim=time_emb_dim)
        self.conv_up4b = ConvBlock(base_channels, base_channels, time_emb_dim=time_emb_dim)
        
        # Output
        self.conv_out = nn.Conv2d(base_channels, img_channels, 1)

        self.time_emb_dim = time_emb_dim

    def forward(self, x, t):
        # time embedding
        t_emb = timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        x0 = self.conv_in1(x, t_emb)       # (B, C, 32, 32)
        x0 = self.conv_in2(x0, t_emb)      # (B, C, 32, 32)

        x1 = self.down1(x0)                # (B, 2C, 16, 16)
        x1 = self.conv_down1a(x1, t_emb)
        x1 = self.conv_down1b(x1, t_emb)

        x2 = self.down2(x1)                # (B, 4C, 8, 8)
        x2 = self.conv_down2a(x2, t_emb)
        x2 = self.conv_down2b(x2, t_emb)

        x3 = self.down3(x2)                # (B, 8C, 4, 4)
        x3 = self.conv_down3a(x3, t_emb)
        x3 = self.conv_down3b(x3, t_emb)

        x4 = self.down4(x3)                # (B, 8C, 2, 2)
        x4 = self.conv_down4a(x4, t_emb)
        x4 = self.conv_down4b(x4, t_emb)

        # Bottleneck
        mid = self.conv_mid1(x4, t_emb)
        mid = self.conv_mid2(mid, t_emb)

        # Decoder
        u1 = self.up1(mid)                 # (B, 8C, 4, 4)
        u1 = torch.cat([u1, x3], dim=1)    # (B, 16C, 4, 4)
        u1 = self.conv_up1a(u1, t_emb)
        u1 = self.conv_up1b(u1, t_emb)

        u2 = self.up2(u1)                  # (B, 4C, 8, 8)
        u2 = torch.cat([u2, x2], dim=1)    # (B, 8C, 8, 8)
        u2 = self.conv_up2a(u2, t_emb)
        u2 = self.conv_up2b(u2, t_emb)

        u3 = self.up3(u2)                  # (B, 2C, 16, 16)
        u3 = torch.cat([u3, x1], dim=1)    # (B, 4C, 16, 16)
        u3 = self.conv_up3a(u3, t_emb)
        u3 = self.conv_up3b(u3, t_emb)

        u4 = self.up4(u3)                  # (B, C, 32, 32)
        u4 = torch.cat([u4, x0], dim=1)    # (B, 2C, 32, 32)
        u4 = self.conv_up4a(u4, t_emb)
        u4 = self.conv_up4b(u4, t_emb)

        out = self.conv_out(u4)
        return out

# DDPM
def make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


def extract(a, t, x_shape):
    out = a.gather(-1, t)  # (B,)
    return out.view(-1, 1, 1, 1).to(dtype=torch.float32)


class DDPM:
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=2e-2, device='cuda'):
        self.model = model
        self.device = device
        self.T = T

        betas = make_beta_schedule(T, beta_start, beta_end).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0
        )

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_om = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_om * noise

    def p_losses(self, x0):
        b = x0.size(0)
        t = torch.randint(0, self.T, (b,), device=self.device).long()
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)
        eps_pred = self.model(x_t, t)
        loss = F.mse_loss(eps_pred, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        One reverse diffusion step: x_t -> x_{t-1}
        Uses the standard DDPM mean parameterization.
        """
        betas_t = extract(self.betas, t, x_t.shape)                        # β_t
        sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, t, x_t.shape) # 1/sqrt(α_t)
        sqrt_one_minus_ac_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )  # sqrt(1 - ᾱ_t)
        posterior_var_t = extract(self.posterior_variance, t, x_t.shape)   # σ_t^2

        # Predict noise ε_θ(x_t, t)
        eps_theta = self.model(x_t, t)

        # μ_θ(x_t, t) = 1/sqrt(α_t) * (x_t - β_t / sqrt(1 - ᾱ_t) * ε_θ)
        model_mean = sqrt_recip_alpha_t * (x_t - betas_t / sqrt_one_minus_ac_t * eps_theta)

        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(-1, 1, 1, 1)  # no noise when t == 0

        x_prev = model_mean + nonzero_mask * torch.sqrt(posterior_var_t) * noise
        return x_prev

    
    @torch.no_grad()
    def sample(self, img_shape, n_samples=64):
        """
        Full reverse diffusion: x_T ~ N(0, I) then T...1.
        img_shape: (C, H, W)
        """
        self.model.eval()
        x_t = torch.randn(n_samples, *img_shape, device=self.device)
        for t in reversed(range(self.T)):
            t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_batch)
        self.model.train()
        return x_t
        
# Main entry
def get_dataloader(batch_size=128, data_root="./data"):
    transform = transforms.Compose([
        # transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    ds = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    return loader


# Training loop
def train_ddpm(cfg):
    device = cfg.device
    dataloader = cfg.dataloader

    # Model and DDPM wrapper
    model = UNet(img_channels=3, base_channels=128, time_emb_dim=256).to(device)
    ddpm = DDPM(model, T=cfg.T, beta_start=cfg.beta_start, beta_end=cfg.beta_end, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)

    img_shape = (3, 32, 32)

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0

        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)  # already normalized to [-1,1]
            optimizer.zero_grad()
            loss = ddpm.p_losses(x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 50 == 0:
                avg_loss = running_loss / 50
                print(f"[DDPM] Epoch {epoch}/{cfg.epochs} Batch {i+1}/{len(dataloader)} Loss={avg_loss:.4f}")
                running_loss = 0.0

        # Sampling & checkpoint logic:
        # - Every 10 epochs: (epoch + 1) % 10 == 0
        # - Every epoch in the last 20 epochs: epoch >= cfg.epochs - 20
        should_sample = ((epoch + 1) % 10 == 0) or (epoch >= cfg.epochs - 20)

        if should_sample:
            # Sampling at end of epoch
            with torch.no_grad():
                samples = ddpm.sample(img_shape=img_shape, n_samples=128).cpu()
                # samples in roughly [-1,1]; rescale to [0,1] for saving
                samples = (samples.clamp(-1, 1) + 1) / 2.0

                epoch_dir = os.path.join(out_dir, f"epoch_{epoch:03d}")
                os.makedirs(epoch_dir, exist_ok=True)

                for j, img in enumerate(samples):
                    img_path = os.path.join(epoch_dir, f"sample_{j:03d}.png")
                    utils.save_image(img, img_path)
        
        # checkpoint
        torch.save(
            {
                "model": model.state_dict(),
                "T": cfg.T,
                "beta_start": cfg.beta_start,
                "beta_end": cfg.beta_end,
            },
            os.path.join(out_dir, f"ckpt_epoch_{epoch:03d}.pt")
        )

    
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--T", type=int, default=1000, help="Number of diffusion steps")
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)
    p.add_argument("--out", type=str, default="./outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def main():
    cfg = parse_args()
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    cfg.device = device

    dataloader = get_dataloader(batch_size=cfg.batch, data_root="../HW_4/data")
    cfg.dataloader = dataloader
    cfg.out_dir = cfg.out

    print("Model: DDPM")
    print("Device:", device)
    print("Diffusion steps T:", cfg.T)
    print("Batches per epoch:", len(dataloader))

    train_ddpm(cfg)

    
if __name__ == "__main__":
    main()