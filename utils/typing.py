from typing import Protocol, Callable, Any

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer


class DatasetLike(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Any: ...


# output, target
MetricF = Callable[[Tensor, Tensor], dict[str, int | float]]

# epoch, model, optimizer, criterion
TrainHookF = Callable[[int, Module, Optimizer, Module], None]
