from torch import Tensor, no_grad
from torch import long, zeros

from .typing import MetricF

from .criterion import MMD, Coral


def bundle(metrics: list[MetricF]) -> MetricF:
    """
    Combines multiple metric functions into a single function.

    Args:
        metrics: List of functions, each with signature (Tensor, Tensor) -> dict.

    Returns:
        A function that takes output and target tensors and returns a merged
        dictionary of all metrics.
    """

    def combined(output: Tensor, target: Tensor) -> dict[str, int | float]:
        result = {}
        for metric in metrics:
            result.update(metric(output, target))
        return result

    return combined


def metrics_with_mask(orig_metrics_fn: MetricF) -> MetricF:
    """
    Wraps a metrics function to ignore masked targets.

    Filters out samples where `y == -1` before calling `orig_metrics_fn`.
    If all samples are masked, returns zeroed metrics with the same keys.

    Args:
        orig_metrics_fn: Callable that computes metrics from (pred, y).

    Returns:
        Wrapped metrics function with masking support.
    """

    def wrapped(pred, y):
        mask = y != -1

        if mask.sum() == 0:
            zeroed = {
                k: 0.0 for k in orig_metrics_fn(zeros(1, pred.size(1)), zeros(1, dtype=long)).keys()
            }
            return zeroed
        return orig_metrics_fn(pred[mask], y[mask])

    return wrapped


def accuracy(output: Tensor, target: Tensor, topk=(1, 5)) -> dict[str, int | float]:
    """
    Computes the top-k accuracy for the specified values of k.

    Args:
        output: Model predictions/logits with shape (batch_size, num_classes)
        target: Ground truth labels with shape (batch_size,)
        topk:   Tuple of integers specifying which top-k accuracies to compute.
                Default: (1, 5) computes top-1 and top-5 accuracy.

    Returns:
        dict: Dict of top-k accuracy scores as percentages (0-100).
              Length matches length of topk parameter.
    """

    with no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[f'acc{k}'] = correct_k.mul_(100.0 / batch_size).item()

    return res


def mmd(output: Tensor, target: Tensor) -> dict[str, int | float]:
    """
    Computes Maximum Mean Discrepancy (MMD) between source and target.

    Splits `output` into source (`target != -1`) and target (`target == -1`)
    subsets. Returns an empty dict if either subset is missing.

    Args:
        output: Model outputs.
        target: Labels with -1 indicating target domain.

    Returns:
        Dict with key 'mmd' and computed value, or empty dict.
    """
    mask = (target != -1).detach()
    has_source = mask.any()
    has_target = (~mask).any()

    if not (has_source and has_target):
        return {}

    mmd_criterion = MMD()

    with no_grad():
        source_d = output[mask]
        target_d = output[~mask]

        res = mmd_criterion(source_d, target_d).item()

    return {'mmd': res}


def coral(output: Tensor, target: Tensor) -> dict[str, int | float]:
    """
    Computes Coral between source and target.

    Splits `output` into source (`target != -1`) and target (`target == -1`)
    subsets. Returns an empty dict if either subset is missing.

    Args:
        output: Model outputs.
        target: Labels with -1 indicating target domain.

    Returns:
        Dict with key 'coral' and computed value, or empty dict.
    """
    mask = (target != -1).detach()
    has_source = mask.any()
    has_target = (~mask).any()

    if not (has_source and has_target):
        return {}

    coral_criterion = Coral()

    with no_grad():
        source_d = output[mask]
        target_d = output[~mask]

        res = coral_criterion(source_d, target_d).item()

    return {'coral': res}
