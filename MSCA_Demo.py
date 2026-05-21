"""
Multi-branch Statistical Channel Attention (MSCA) demo.

This script provides a standalone demonstration of the MSCA module used in BSCFNet.
It is designed for module-level illustration only and is not the full BSCFNet pipeline.

Default setting:
    branches = 4

Input:
    x: torch.Tensor, shape [B, C, N, K]

Output:
    out: torch.Tensor, shape [B, C, N, K]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiBranchStatisticalChannelAttention(nn.Module):
    def __init__(self, channels: int, branches: int = 4, reduction: int = 4):
        super().__init__()
        if branches <= 0:
            raise ValueError("branches should be a positive integer.")
        if channels <= 0:
            raise ValueError("channels should be a positive integer.")
        hidden = max(channels // reduction, 1)
        self.branches = branches
        self.excitation_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * channels, hidden, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            for _ in range(branches)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x should have shape [B, C, N, K].")
        mean = torch.mean(x, dim=3, keepdim=True)  # [B, C, N, 1]
        std = torch.std(x, dim=3, keepdim=True, unbiased=False)  # [B, C, N, 1]
        stat = torch.cat([mean, std], dim=1)  # [B, 2C, N, 1]
        weights = [branch(stat) for branch in self.excitation_branches]
        attention = torch.stack(weights, dim=0).mean(dim=0)  # [B, C, N, 1]
        return x * attention  # broadcast along K


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, N, K = 2, 32, 1024, 24
    x = torch.randn(B, C, N, K)
    msca = MultiBranchStatisticalChannelAttention(channels=C, branches=4)
    y = msca(x)
    print("MSCA demo")
    print("Input:", x.shape)
    print("Output:", y.shape)
