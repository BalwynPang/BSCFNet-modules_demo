"""
Global Context Extractor (GCE) demo.

This script provides a standalone demonstration of the GCE module used in BSCFNet.
It is designed for module-level illustration only and is not the full BSCFNet pipeline.

Key setting:
    GAP is used for global context aggregation.

Input:
    xyz: torch.Tensor, shape [B, N, 3]

Output:
    global_feature: torch.Tensor, shape [B, global_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GlobalContextExtractor(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, global_dim: int = 128):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, global_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim != 3:
            raise ValueError("xyz should have shape [B, N, 3].")
        point_features = self.point_mlp(xyz)        # [B, N, global_dim]
        global_feature = torch.mean(point_features, dim=1)  # GAP: [B, global_dim]
        return global_feature


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N = 2, 4096
    xyz = torch.randn(B, N, 3)
    gce = GlobalContextExtractor(input_dim=3, hidden_dim=64, global_dim=128)
    g = gce(xyz)
    print("GCE demo")
    print("Input xyz:", xyz.shape)
    print("Global feature:", g.shape)
