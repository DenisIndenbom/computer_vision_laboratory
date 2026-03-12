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

        combined_rows = combined.unsqueeze(0).expand(
            n_total, n_total, combined.size(1)
        )
        combined_cols = combined.unsqueeze(1).expand(
            n_total, n_total, combined.size(1)
        )
        squared_distances = ((combined_rows - combined_cols) ** 2).sum(dim=2)

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(squared_distances.data) / \
                (n_total ** 2 - n_total)

        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)

        bandwidth_scales = [
            bandwidth * (self.kernel_mul ** i)
            for i in range(self.kernel_num)
        ]

        kernel_matrices = [
            torch.exp(-squared_distances / bandwidth_scale)
            for bandwidth_scale in bandwidth_scales
        ]

        return sum(kernel_matrices, start=torch.ones_like(kernel_matrices[0]))

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        batch_size = source.size()[0]

        kernels = self._gaussian_kernel(source, target)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        return torch.mean(XX + YY - XY - YX)
