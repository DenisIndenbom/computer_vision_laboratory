from typing import Iterator

import torch
from torch.utils.data import Sampler


class DomainBatchSampler(Sampler):
    """
    Batch sampler that yields balanced batches from source and target domains.

    Each batch contains `batch_size // 2` samples from source and the same number from target.
    """

    def __init__(self, source_size: int, target_size: int, batch_size: int, shuffle: bool = True):
        assert batch_size % 2 == 0, 'batch_size must be even'
        assert source_size >= batch_size // 2, 'source_size too small for domain_batch_size'
        assert target_size >= batch_size // 2, 'target_size too small for domain_batch_size'

        self.source_size = source_size
        self.target_size = target_size
        self.batch_size = batch_size
        self.domain_batch_size = batch_size // 2
        self.shuffle = shuffle

        # Source and target indices
        self.source_indices = torch.arange(source_size, dtype=torch.long)
        self.target_indices = torch.arange(source_size, source_size + target_size, dtype=torch.long)

        # Number of full batches we can form
        self.num_source_batches = source_size // self.domain_batch_size
        self.num_target_batches = target_size // self.domain_batch_size

    def __iter__(self) -> Iterator[list[int]]:
        # Shuffle indices if required
        if self.shuffle:
            source_perm = torch.randperm(self.source_size)
            target_perm = torch.randperm(self.target_size)
            source_idx = self.source_indices[source_perm]
            target_idx = self.target_indices[target_perm]
        else:
            source_idx = self.source_indices
            target_idx = self.target_indices

        # Trim to aligned sizes (drop remainder)
        source_idx = source_idx[: self.num_source_batches * self.domain_batch_size]
        target_idx = target_idx[: self.num_target_batches * self.domain_batch_size]

        # Reshape into batches: (num_source_batches, domain_batch_size)
        source_batches = source_idx.view(self.num_source_batches, self.domain_batch_size)
        target_batches = target_idx.view(self.num_target_batches, self.domain_batch_size)

        # Build a list of (source_batch, target_batch) pairs
        for i in range(self.num_source_batches):
            src_batch = source_batches[i].tolist()
            tgt_batch = target_batches[i % self.num_target_batches].tolist()

            yield src_batch + tgt_batch

    def __len__(self) -> int:
        return self.num_source_batches
