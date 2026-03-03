from torch import no_grad
from torch import Tensor


def accuracy(output: Tensor, target: Tensor, topk=(1, 5)) -> dict[str, float]:
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
            res[f'acc{k}'] = correct_k.mul_(100.0 / batch_size)

    return res
