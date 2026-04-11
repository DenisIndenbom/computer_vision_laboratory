import random
import numpy as np

from os import path, environ

import torch
import torch.nn as nn

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from .typing import MetricF, TrainHookF

from .history import History


def _add_scalars(
    writer: SummaryWriter | None, tag: str, scalars: dict[str, int | float], step: int
):
    """
    Add scalars in tensorboard SummaryWriter.

    Args:
        writer: Tensorboard summary writer.
        tag: Tag of category.
        scalars: dict of scalar values
        step: Step value to record
    """
    if not writer:
        return

    for name, value in scalars.items():
        writer.add_scalar(f'{tag}/{name}', value, step, new_style=True)


def _get_np_state():
    state = list(np.random.get_state())
    state[1] = state[1].tolist()  # type: ignore
    return tuple(state)


def _set_seed(seed: int):
    """
    Set seed in the pytorch framework.

    Args:
        seed: Seed to set.
    """
    # Common libs
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_random_state(filepath: str):
    """
    Save the current random states of PyTorch (CPU and CUDA) and the random module
    to a file using torch.save.

    Args:
        filepath: Destination file path (e.g., "random_state.pt").
    """
    state = {
        'torch': torch.random.get_rng_state(),  # CPU state
        'random': random.getstate(),  # Python's random state
        'numpy': _get_np_state(),  # Numpy's random state
    }

    if torch.cuda.is_available():
        # state of all CUDA devices
        state['torch_cuda'] = torch.cuda.get_rng_state_all()

    torch.save(state, filepath)


def _load_random_state(filepath: str):
    """
    Load random states from a file and restore them.

    Args:
        filepath: Path to the file previously created by save_random_state().
    """
    if not path.exists(filepath):
        return

    state = torch.load(filepath)

    torch.random.set_rng_state(state['torch'])
    random.setstate(state['random'])
    np.random.set_state(state['numpy'])

    if 'torch_cuda' in state:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


class tqdmd(tqdm):
    """
    tqdm with docker support.
    """

    in_docker = environ.get('RUNNING_IN_DOCKER', False)

    def update(self, n=1):
        result = super().update(n)

        if result and self.in_docker:
            print(f'{self}\n')

        return result

    def display(self, msg=None, pos=None):
        if self.in_docker:
            return

        return super().display(msg, pos)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    metrics: MetricF,
    epochs: int,
    start_epoch: int = 0,
    checkpoint_interval: int = 5,
    checkpoint_path: str = '.',
    device: torch.device = torch.device('cpu'),
    seed: int | None = None,
    verbose: bool = True,
    summary_writer: SummaryWriter | None = None,
    post_epoch_hook: TrainHookF | None = None,
) -> dict[str, list[int | float | None]]:
    """
    Train the classification model for a given number of epochs, evaluating on the validation set after each epoch.

    Args:
        model: Neural network model.
        train_loader: DataLoader supplying training batches.
        val_loader: DataLoader supplying validation batches.
        optimizer: Optimizer for updating model weights.
        criterion: Loss function.
        metrics: Function that takes predictions and targets and returns a dict of metric names to values.
        epochs: Number of training epochs.
        start_epoch: Epoch number to resume training from.
        checkpoint_interval: Interval of epochs for saving the model
        checkpoint_path: Path for saving the model
        device: Device on which to perform computation.
        seed: Random seed for training reproducibility.
        verbose: If True, print per-epoch metrics.
        summary_writer: Tensorboard summary writer.
        post_epoch_hook: A hook function that is called at the end of each training epoch.

    Returns:
        Dictionary containing lists of metrics for each epoch.
        Always includes 'train_loss' and 'val_loss'.
        Additional keys are prefixed with 'train_' and 'val_'
        corresponding to the metric names returned by the `metrics` function.

        For example, if `metrics` returns {'acc': 0.95, 'f1': 0.92},
        the history will contain 'train_acc', 'val_acc', 'train_f1', 'val_f1'.
        Each list has length equal to the number of epochs trained.
    """
    # History storage
    history = History(start_epoch + 1)

    # Store model name
    model_name = type(model).__name__

    # Move model to device
    model.to(device)

    # Set seed if specified
    if seed is not None:
        _set_seed(seed)

    # Load checkpoint if specified
    if start_epoch > 0:
        print('Loading checkpoint...', end='')
        model.load_state_dict(
            torch.load(path.join(checkpoint_path, f'{model_name}_checkpoint_{start_epoch}.model'))
        )
        optimizer.load_state_dict(
            torch.load(path.join(checkpoint_path, f'{model_name}_checkpoint_{start_epoch}.optim'))
        )
        _load_random_state(path.join(checkpoint_path, f'{model_name}_checkpoint_{start_epoch}.rng'))
        history.load(path.join(checkpoint_path, f'{model_name}_checkpoint_{start_epoch}.metrics'))
        print('OK')

    for epoch in range(start_epoch + 1, epochs + 1):
        # --- Training ---
        model.train()

        train_loop = tqdmd(
            train_loader, desc=f'Epoch {epoch}/{epochs} [Train]', leave=False, disable=not verbose
        )
        for x_batch, y_batch in train_loop:
            # Zero gradient
            optimizer.zero_grad()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            pred = model(x_batch)

            # Backward pass
            loss = criterion(pred, y_batch)
            loss.backward()

            # Gradient step
            optimizer.step()

            # Update metrics
            batch_metrics = metrics(pred, y_batch)
            history.add_batch(batch_metrics)
            history.add_batch({'loss': loss.item()})

            # Update tqdm postfix
            train_loop.set_postfix(loss=loss.item())

        # Average train batches and log to history
        avg_train_metrics = history.commit(prefix='train_')
        # Log averaged metrics to tensorboard
        _add_scalars(summary_writer, 'train', avg_train_metrics, epoch)

        # --- Validation ---
        model.eval()

        val_loop = tqdmd(
            val_loader, desc=f'Epoch {epoch}/{epochs} [Val]', leave=False, disable=not verbose
        )
        with torch.no_grad():
            for x_batch, y_batch in val_loop:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward pass
                pred = model(x_batch)
                loss = criterion(pred, y_batch)

                # Update metrics
                batch_metrics = metrics(pred, y_batch)
                history.add_batch(batch_metrics)
                history.add_batch({'loss': loss.item()})

                # Update tqdm postfix
                val_loop.set_postfix(loss=loss.item())

        # Average validation batches and log to history
        avg_val_metrics = history.commit(prefix='val_')
        # Log averaged metrics to tensorboard
        _add_scalars(summary_writer, 'validation', avg_val_metrics, epoch)

        if post_epoch_hook is not None:
            post_epoch_hook(epoch, model, optimizer, criterion)

        # Print epoch summary if verbose
        if verbose:
            train_metric_str = ' | '.join(
                [f'Train {name}: {value:.4f}' for name, value in avg_train_metrics.items()]
            )
            val_metric_str = ' | '.join(
                [f'Val {name}: {value:.4f}' for name, value in avg_val_metrics.items()]
            )
            print(f'Epoch {epoch:3d} | {train_metric_str} | {val_metric_str}')

        # Save checkpoint if interval
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            print('Saving checkpoint...', end='')
            torch.save(
                model.state_dict(),
                path.join(checkpoint_path, f'{model_name}_checkpoint_{epoch}.model'),
            )
            torch.save(
                optimizer.state_dict(),
                path.join(checkpoint_path, f'{model_name}_checkpoint_{epoch}.optim'),
            )
            _save_random_state(path.join(checkpoint_path, f'{model_name}_checkpoint_{epoch}.rng'))
            history.dump(path.join(checkpoint_path, f'{model_name}_checkpoint_{epoch}.metrics'))
            print('OK')

        if summary_writer:
            summary_writer.flush()

        history.next_epoch()

    return history.to_dict()


def test(
    model: nn.Module,
    dataloader: DataLoader,
    metrics: MetricF,
    device: torch.device = torch.device('cpu'),
) -> dict[str, float]:
    """
    Evaluate the classification model on a test set.

    Args:
        model: Neural network model.
        dataloader: DataLoader supplying test batches.
        metrics: Function that takes predictions and targets and returns a dict of metric names to values.
        device: Device on which to perform computation.

    Returns:
        Dictionary containing average metrics on the test set (e.g., 'acc1', 'acc5').
    """
    model.eval()
    total_metrics: dict[str, float] = {}

    test_loop = tqdmd(dataloader, desc='[Test]', leave=False)
    with torch.no_grad():
        for x_batch, y_batch in test_loop:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            pred = model(x_batch)

            # Update metrics
            batch_metrics = metrics(pred, y_batch)
            for name, value in batch_metrics.items():
                total_metrics[name] = total_metrics.get(name, 0.0) + value

            test_loop.set_postfix(batch_metrics)

    return {name: total / len(dataloader) for name, total in total_metrics.items()}
