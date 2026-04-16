from typing import Iterator

from torch import Generator, rand_like
from torch.autograd import Function
from torch.nn import Parameter
from torch.utils.hooks import RemovableHandle


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # pyright: ignore[reportIncompatibleMethodOverride]
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    """
    Gradient reversal layer for adversarial training.

    During the forward pass, this acts as an identity function. During
    backpropagation, the gradients are multiplied by ``-lambda_``, effectively
    reversing the gradient direction and scaling it.

    Args:
        x: Input tensor.
        lambda_: Gradient scaling factor. Defaults to 1.0.

    Returns:
        Tensor with the same value as ``x``, but with reversed gradients
        during the backward pass.
    """
    return GradientReversalFunction.apply(x, lambda_)


def freeze_params(
    params: Iterator[tuple[str, Parameter]],
    fraction: float = 0.2,
    seed: int = 42,
    exclude: list[str] | None = None,
) -> dict[str, RemovableHandle]:
    """
    Randomly choose weights for zeroing the gradient during whole training.

    Args:
        params: Model parameters to freeze.
        fraction: Ratio of freezed parameters. Defaults to 0.2.
        seed: Random seed. Defaults to 42.
        exclude: List of parameters to exclude from freezing. Defaults to None.

    Returns:
        Dictionary of removable handles for each parameter.
    """

    g = Generator().manual_seed(seed)
    hooks: dict[str, RemovableHandle] = {}

    for name, param in params:
        if not param.requires_grad:
            continue

        if exclude is not None and name in exclude:
            continue

        mask = (rand_like(param, generator=g) > fraction).float()

        def make_hook(mask):
            def hook(grad):
                return grad * mask

            return hook

        hooks[name] = param.register_hook(make_hook(mask))

    return hooks
