import torch
import torch.nn as nn


class MMD(nn.Module):
    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5, fix_sigma: float | None = None):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def _gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes multi-scale RBF kernel matrix.
        """
        total = torch.cat([source, target], dim=0)  # [n+m, d]

        # Pairwise squared distances
        dist = torch.cdist(total, total, p=2) ** 2  # [n+m, n+m]

        # Bandwidth estimation
        if self.fix_sigma is not None:
            bandwidth = self.fix_sigma
        else:
            mask = ~torch.eye(dist.size(0), dtype=torch.bool,
                              device=dist.device)
            bandwidth = dist[mask].median()

            # Fallback if median is bad
            if bandwidth < 1e-8:
                bandwidth = dist[mask].mean()

            # Stabilize
            bandwidth = bandwidth.detach()
            bandwidth = bandwidth.clamp(min=1e-8)

            # Adjust for multi-kernel
            bandwidth /= self.kernel_mul ** (self.kernel_num // 2)

        # multi-scale kernels
        bandwidths = [
            bandwidth * (self.kernel_mul ** i)
            for i in range(self.kernel_num)
        ]

        kernel = torch.zeros_like(dist)
        for bw in bandwidths:
            kernel += torch.exp(-dist / bw)

        return kernel

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes unbiased MMD^2 between source and target.
        """
        n = source.size(0)
        m = target.size(0)

        kernel = self._gaussian_kernel(source, target)

        XX = kernel[:n, :n]
        YY = kernel[n:n + m, n:n + m]
        XY = kernel[:n, n:n + m]

        return XX.mean() + YY.mean() - 2 * XY.mean()


class Coral(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d = source.size(1)

        # Calculate the covariance matrices
        source_c = self._covariance(source)
        target_c = self._covariance(target)

        # Calculate the difference between matrices
        diff = (source_c - target_c)

        return (diff * diff).sum() / (4 * d * d)

    def _covariance(self, x: torch.Tensor) -> torch.Tensor:
        # Center the data around the origin
        x = x - x.mean(dim=0, keepdim=True)
        # Covariance matrix - X^T * X / (N-1)
        return x.T @ x / (x.size(0) - 1)
