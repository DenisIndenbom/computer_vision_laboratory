import importlib
import pkgutil

from typing import Callable, Any, TypedDict


class TrainArgs(TypedDict):
    epochs: int
    start_epoch: int
    checkpoint_interval: int
    checkpoint_path: str
    verbose: bool


MethodType = Callable[[TrainArgs], Any]

METHOD_REGISTRY: dict[str, MethodType] = {}


def register(name: str) -> Callable[[MethodType], MethodType]:
    def decorator(fn: MethodType) -> MethodType:
        METHOD_REGISTRY[name] = fn
        return fn

    return decorator


for _, module_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f'{__name__}.{module_name}')
