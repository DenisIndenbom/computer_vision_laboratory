from torch import Tensor, long, no_grad, zeros

from .typing import ConditionF, CriterionF, MetricF, TransformF


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


def with_mask(orig_metrics_fn: MetricF, label: int = -1) -> MetricF:
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


def with_slice(orig_metrics_fn: MetricF, index: int) -> MetricF:
    """
    Wraps a metrics function to compute only on a specific slice of predictions.

    Args:
        orig_metrics_fn: Callable that computes metrics from `(pred, y)`.
        index: Index to slice from the first dimension of predictions.

    Returns:
        Wrapped metrics function.
    """

    def wrapped(pred, y):
        return orig_metrics_fn(pred[index], y)

    return wrapped


def with_kwargs(orig_metrics_fn: MetricF, **kwargs) -> MetricF:
    """
    Wraps a metrics function to forward specified keyword arguments.

    Args:
        orig_metrics_fn: Callable that computes metrics from `(pred, y)` and
                         possibly additional keyword arguments.
        **kwargs: Keyword arguments to forward to the metrics function.

    Returns:
        Wrapped metrics function.
    """

    def wrapped(pred, y):
        return orig_metrics_fn(pred, y, **kwargs)

    return wrapped


def with_prefix(metric_fn: MetricF, prefix: str) -> MetricF:
    """
    Wraps a metrics function to prepend a prefix to all returned metric keys.

    Args:
        metric_fn: Metric function that returns a dictionary of named metrics.
        prefix: String to prepend to each key (e.g., "train_", "val/").

    Returns:
        Wrapped metrics function that returns a dictionary with prefixed keys.
    """

    def wrapped(pred, target):
        return {f'{prefix}{k}': v for k, v in metric_fn(pred, target).items()}

    return wrapped


def apply_if(metric_fn: MetricF, condition: ConditionF) -> MetricF:
    """
    Wraps a metrics function to be executed only when a condition is met.

    Args:
        metric_fn: Metric function to be conditionally applied.
        condition: Callable that takes `(pred, target)` and returns a boolean.
            If `True`, `metric_fn` is called; otherwise an empty dict is returned.

    Returns:
        Wrapped metrics function that respects the condition.
    """

    def wrapped(pred, target):
        if condition(pred, target):
            return metric_fn(pred, target)
        return {}

    return wrapped


def with_transform(metric_fn: MetricF, transform: TransformF) -> MetricF:
    """
    Wraps a metrics function to apply a preprocessing transform to pred and target.

    Args:
        metric_fn: Metric function to call with transformed inputs.
        transform: Callable that takes `(pred, target)` and returns a tuple
                   `(transformed_pred, transformed_target)`.

    Returns:
        Wrapped metrics function that preprocesses the inputs before calling `metric_fn`.
    """

    def wrapped(pred, target):
        pred_t, target_t = transform(pred, target)
        return metric_fn(pred_t, target_t)

    return wrapped


def distance_metric(criterion: CriterionF, name: str) -> MetricF:
    """
    Creates a metric that computes a distance between two masked groups.

    The function splits the output tensor into two groups based on the target
    mask: samples where `target != -1` form the "source" group, and samples
    where `target == -1` form the "target" group. It then applies the given
    criterion (e.g., a distance or loss) between these two groups.

    If one of the groups is empty, the metric returns an empty dictionary.

    Args:
        criterion: A callable that takes two tensors `(source, target)` and
            returns a scalar distance/loss (e.g., torch.cdist, F.mse_loss).
        name: The key under which the computed value will be stored in the
            returned dictionary.

    Returns:
        A metric function that returns a dict `{name: distance_value}`.
    """

    def metric_fn(output: Tensor, target: Tensor) -> dict[str, float]:
        mask = target != -1
        has_source = mask.any()
        has_target = (~mask).any()

        if not (has_source and has_target):
            return {}

        with no_grad():
            value = criterion(output[mask], output[~mask]).item()

        return {name: value}

    return metric_fn


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


def binary_accuracy(output: Tensor, target: Tensor) -> dict[str, float]:
    """
    Computes accuracy for binary classification.

    Supports two common output formats:
    - Single logit per sample (shape: (batch_size,) or (batch_size, 1)):
      uses sigmoid and threshold 0.5.
    - Two logits per sample (shape: (batch_size, 2)):
      uses argmax (equivalent to softmax).

    Args:
        output: Model predictions/logits. Either shape (batch_size,), (batch_size, 1),
                or (batch_size, 2).
        target: Ground truth labels with shape (batch_size,), containing 0 or 1.

    Returns:
        dict: Dict with key 'acc' and value as accuracy percentage (0-100).
    """

    with no_grad():
        batch_size = target.size(0)

        if output.ndim == 2 and output.size(1) == 2:
            pred = output.argmax(dim=1)
        else:
            out = output.squeeze(-1)
            pred = (out.sigmoid() > 0.5).long()

        target = target.long()

        correct = pred.eq(target).float().sum()
        acc = correct.mul_(100.0 / batch_size).item()

    return {'acc': acc}
