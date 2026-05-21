"""
Boundary-Enhanced Hybrid Sampling (BEHS) demo.

This script provides a standalone demonstration of the BEHS strategy used in BSCFNet.
It is designed for module-level illustration only and is not the full BSCFNet pipeline.

Default settings:
    KNN = 24
    rho = 0.5

Input:
    xyz: torch.Tensor, shape [B, N, 3]
    features: torch.Tensor, shape [B, C, N, 1]

Output:
    sampled_features: torch.Tensor, shape [B, C, N_sampled, 1]
    sampled_indices: torch.Tensor, shape [B, N_sampled]
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree


class BoundaryEnhancedHybridSampling:
    def __init__(self, knn: int = 24, rho: float = 0.5, use_height_filter: bool = True):
        if knn <= 3:
            raise ValueError("knn should be larger than 3 for stable normal estimation.")
        if not (0.0 < rho < 1.0):
            raise ValueError("rho should be in (0, 1).")
        self.knn = knn
        self.rho = rho
        self.use_height_filter = use_height_filter

    @staticmethod
    def _compute_weld_coordinate_system(points: np.ndarray):
        pca = PCA(n_components=3)
        pca.fit(points)
        main_axis = pca.components_[0]
        height_axis = pca.components_[2]
        cross_axis = np.cross(main_axis, height_axis)
        min_z_point = points[np.argmin(points[:, 2])]
        centroid = points.mean(axis=0)
        if np.dot(height_axis, centroid - min_z_point) < 0:
            height_axis *= -1.0
        return main_axis, height_axis, cross_axis, min_z_point

    def _compute_normals(self, points: np.ndarray):
        tree = KDTree(points)
        _, idx = tree.query(points, k=self.knn)
        neighbors = points[idx]
        centroids = neighbors.mean(axis=1, keepdims=True)
        cov_matrices = np.einsum("nki,nkj->nij", neighbors - centroids, neighbors - centroids) / float(self.knn)
        _, eigenvectors = np.linalg.eigh(cov_matrices)
        normals = eigenvectors[:, :, 0]
        mean_normal = normals.mean(axis=0)
        mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-12)
        flip_mask = np.dot(normals, mean_normal) < 0
        normals[flip_mask] *= -1.0
        return normals, idx

    def _compute_normal_variation(self, points, normals, neighbor_idx, height_axis, reference_point):
        neighbor_normals = normals[neighbor_idx]
        cos_values = np.einsum("nkj,nj->nk", neighbor_normals, normals)
        cos_values = np.clip(np.abs(cos_values), 0.0, 1.0)
        normal_variation = np.mean(np.arccos(cos_values), axis=1)
        if self.use_height_filter:
            projected_height = np.dot(points - reference_point, height_axis)
            low, high = np.percentile(projected_height, [5, 95])
            mask = (projected_height >= low) & (projected_height <= high)
            normal_variation[~mask] = 0.0
        return normal_variation

    def __call__(self, xyz: torch.Tensor, features: torch.Tensor, num_sampled: int | None = None):
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError("xyz should have shape [B, N, 3].")
        if features.ndim != 4:
            raise ValueError("features should have shape [B, C, N, 1].")
        if xyz.shape[0] != features.shape[0] or xyz.shape[1] != features.shape[2]:
            raise ValueError("xyz and features have inconsistent batch or point dimensions.")

        batch_size, num_points, _ = xyz.shape
        if num_sampled is None:
            num_sampled = max(1, num_points // 4)
        if num_sampled > num_points:
            raise ValueError("num_sampled should not exceed the number of input points.")

        sampled_features_list = []
        sampled_indices_list = []
        for b in range(batch_size):
            xyz_np = xyz[b].detach().cpu().numpy()
            _, height_axis, _, reference_point = self._compute_weld_coordinate_system(xyz_np)
            normals, neighbor_idx = self._compute_normals(xyz_np)
            scores = self._compute_normal_variation(xyz_np, normals, neighbor_idx, height_axis, reference_point)

            num_edge = int(num_sampled * self.rho)
            num_random = num_sampled - num_edge
            edge_indices = np.argpartition(scores, -num_edge)[-num_edge:]
            mask = np.ones(num_points, dtype=bool)
            mask[edge_indices] = False
            remaining = np.where(mask)[0]
            if len(remaining) < num_random:
                random_indices = remaining
            else:
                random_indices = np.random.choice(remaining, size=num_random, replace=False)
            sampled_indices_np = np.concatenate([edge_indices, random_indices])
            np.random.shuffle(sampled_indices_np)

            sampled_indices = torch.tensor(sampled_indices_np, dtype=torch.long, device=xyz.device)
            sampled_feature = torch.index_select(features[b, :, :, 0], dim=1, index=sampled_indices).unsqueeze(-1)
            sampled_features_list.append(sampled_feature.unsqueeze(0))
            sampled_indices_list.append(sampled_indices.unsqueeze(0))

        sampled_features = torch.cat(sampled_features_list, dim=0)
        sampled_indices = torch.cat(sampled_indices_list, dim=0)
        return sampled_features, sampled_indices


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    B, N, C = 2, 4096, 16
    xyz = torch.randn(B, N, 3)
    features = torch.randn(B, C, N, 1)
    behs = BoundaryEnhancedHybridSampling(knn=24, rho=0.5)
    sampled_features, sampled_indices = behs(xyz, features, num_sampled=1024)
    print("BEHS demo")
    print("Input xyz:", xyz.shape)
    print("Input features:", features.shape)
    print("Sampled indices:", sampled_indices.shape)
    print("Sampled features:", sampled_features.shape)
