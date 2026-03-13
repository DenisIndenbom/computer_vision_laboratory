import torch
import torch.nn as nn


class MMD(nn.Module):
    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5, fix_sigma: float | None = None):
        super(MMD, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def _gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_total = source.size(0) + target.size(0)
        combined = torch.cat([source, target], dim=0)

        squared_distances = torch.cdist(combined, combined, p=2) ** 2

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            off_diag_sum = squared_distances.sum() - squared_distances.diag().sum()
            bandwidth = off_diag_sum / (n_total * (n_total - 1))
            bandwidth = bandwidth.detach()
            bandwidth /= self.kernel_mul ** (self.kernel_num // 2)

        bandwidths = [bandwidth * (self.kernel_mul ** i)
                      for i in range(self.kernel_num)]

        kernel = torch.zeros_like(squared_distances)
        for bw in bandwidths:
            kernel += torch.exp(-squared_distances / bw)

        return kernel

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kernel = self._gaussian_kernel(source, target)

        n = source.size(0)
        XX = kernel[:n, :n]
        YY = kernel[n:, n:]
        XY = kernel[:n, n:]

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

        return torch.mean(torch.mul(diff, diff)) / (4 * d * d)

    def _covariance(self, x: torch.Tensor) -> torch.Tensor:
        # Center the data around the origin
        x = x - x.mean(dim=0, keepdim=True)
        # Covariance matrix - X^T * X / (N-1)
        return x.T @ x / (x.size(0) - 1)
