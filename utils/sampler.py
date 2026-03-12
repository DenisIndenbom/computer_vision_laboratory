import random

from itertools import batched

from torch.utils.data import Sampler


class DomainBatchSampler(Sampler):
    def __init__(self, source_size: int, target_size: int, batch_size: int):
        assert source_size > target_size

        self.source_indices = list(range(source_size))
        self.target_indices = list(
            range(source_size, source_size + target_size))

        self.domain_batch_size = batch_size // 2

        self.source_align = (
            source_size // self.domain_batch_size) * self.domain_batch_size
        self.target_align = (
            target_size // self.domain_batch_size) * self.domain_batch_size

    def __iter__(self):
        random.shuffle(self.source_indices)
        random.shuffle(self.target_indices)

        target_iter = batched(
            self.target_indices[:self.target_align], self.domain_batch_size)

        for source_batch in batched(self.source_indices[:self.source_align], self.domain_batch_size):
            try:
                target_batch = next(target_iter)
            except StopIteration:
                random.shuffle(self.target_indices)
                target_iter = batched(
                    self.target_indices[:self.target_align], self.domain_batch_size)
                target_batch = next(target_iter)

            yield source_batch + target_batch

    def __len__(self):
        return len(self.source_indices) // self.domain_batch_size
