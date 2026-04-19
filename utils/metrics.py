from torch import Tensor, long, no_grad, zeros

from .criterion import MMD, Coral
from .typing import MetricF


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


def metrics_with_mask(orig_metrics_fn: MetricF, label: int = -1) -> MetricF:
    """
    Wraps a metrics function to ignore masked targets.

    Filters out samples where `y == label` before calling `orig_metrics_fn`.
    If all samples are masked, returns zeroed metrics with the same keys.

    Args:
        orig_metrics_fn: Callable that computes metrics from (pred, y).
        label: Target value to treat as masked and ignore during metric computation. Defaults to -1.

    Returns:
        Wrapped metrics function with masking support.
    """

    def wrapped(pred, y):
        mask = y != label

        if mask.sum() == 0:
            zeroed = {
                k: 0.0 for k in orig_metrics_fn(zeros(1, pred.size(1)), zeros(1, dtype=long)).keys()
            }
            return zeroed
        return orig_metrics_fn(pred[mask], y[mask])

    return wrapped


def metrics_with_slice(orig_metrics_fn: MetricF, index: int) -> MetricF:
    """
    Wraps a metrics function to compute only on a specific slice of predictions.

    Args:
        orig_metrics_fn: Callable that computes metrics from (pred, y).
        index: Index to slice from the first dimension of predictions.

    Returns:
        Wrapped metrics function.
    """

    def wrapped(pred, y):
        return orig_metrics_fn(pred[index], y)

    return wrapped


def accuracy_at(topk=(1, 5), prefix: str = '') -> MetricF:
    """
    Wraps accuracy with preset topk and prefix.

    Args:
        topk: Tuple of integers specifying which top-k accuracies to compute.. Defaults to (1, 5).
        prefix: Optional string to prepend to the returned metric keys. Defaults to ''.
    """

    def wrapped(*args):
        return accuracy(*args, topk=topk, prefix=prefix)

    return wrapped


def accuracy(
    output: Tensor, target: Tensor, topk=(1, 5), prefix: str = ''
) -> dict[str, int | float]:
    """
    Computes the top-k accuracy for the specified values of k.

    Args:
        output: Model predictions/logits with shape (batch_size, num_classes)
        target: Ground truth labels with shape (batch_size,)
        topk:   Tuple of integers specifying which top-k accuracies to compute.
                Default: (1, 5) computes top-1 and top-5 accuracy.
        prefix: Optional string to prepend to the returned metric keys.
                For example, prefix='val_' yields keys like 'val_acc1', 'val_acc5'.

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
            res[f'{prefix}acc{k}'] = correct_k.mul_(100.0 / batch_size).item()

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
