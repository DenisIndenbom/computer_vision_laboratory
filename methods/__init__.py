import importlib
import pkgutil

from typing import Callable, Any, TypedDict


class TrainArgs(TypedDict):
    name: str
    data: str
    epochs: int
    start_epoch: int
    batch_size: int
    learning_rate: float
    workers: int
    seed: int
    checkpoint_path: str
    checkpoint_interval: int
    device: str
    verbose: bool


MethodType = Callable[[TrainArgs], Any]

METHOD_REGISTRY: dict[str, MethodType] = {}


def register(name: str) -> Callable[[MethodType], MethodType]:
    """
    Register train method in METHOD_REGISTRY for train script

    Args:
        name: Method name.

    Returns:
        Method entrypoint.
    """
    def decorator(fn: MethodType) -> MethodType:
        METHOD_REGISTRY[name] = fn
        return fn

    return decorator


# load all method modules
for _, module_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f'{__name__}.{module_name}')
