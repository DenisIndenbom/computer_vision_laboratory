from typing import Any, Callable, Protocol

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer


class DatasetLike(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Any: ...


# output, target
CriterionF = Callable[[Tensor, Tensor], Tensor]

# output, target
MetricF = Callable[[Tensor, Tensor], dict[str, int | float]]

# output, target
ConditionF = Callable[[Tensor, Tensor], bool]
TransformF = Callable[[Any, Any], tuple[Any, Any]]

# epoch, model, optimizer, criterion
TrainHookF = Callable[[int, Module, Optimizer, CriterionF], None]
