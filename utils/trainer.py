import json

from os import path
from typing import Callable

import torch
import torch.nn as nn

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .history import History


def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        metrics: Callable[[torch.Tensor, torch.Tensor], dict[str, float]],
        epochs: int,
        start_epoch: int = 0,
        checkpoint_interval: int = 5,
        checkpoint_path: str = '.',
        device: torch.device = torch.device('cpu'),
        verbose: bool = True) -> dict[str, list[int | float | None]]:
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
        verbose: If True, print per-epoch metrics.

    Returns:
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
    history = History(start_epoch if start_epoch > 0 else 1)

    # Store model name
    model_name = type(model).__name__

    # Move model to device
    model.to(device)

    # Load checkpoint if specified
    if start_epoch > 0:
        print('Loading checkpoint...', end='')
        model.load_state_dict(
            torch.load(path.join(checkpoint_path, f'{model_name}_checkpoint_{start_epoch}.model')))
        optimizer.load_state_dict(
            torch.load(path.join(checkpoint_path, f'{model_name}_checkpoint_{start_epoch}.optim')))
        history.load(path.join(checkpoint_path,
                     f'{model_name}_checkpoint_{start_epoch}.metrics'))
        print('OK')

    for epoch in range(start_epoch + 1, epochs + 1):
        # --- Training ---
        model.train()

        total_train_loss = 0.0
        total_train_metrics: dict[str, float] = {}

        train_loop = tqdm(
            train_loader,
            desc=f'Epoch {epoch}/{epochs} [Train]',
            leave=False
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
            for name, value in batch_metrics.items():
                total_train_metrics[name] = \
                    total_train_metrics.get(name, 0.0) + value
            total_train_loss += loss.item()

            # Update tqdm postfix
            train_loop.set_postfix(loss=loss.item())

        # Compute average by train batch
        num_train_batches = len(train_loader)
        avg_train_metrics = {
            name: total / num_train_batches
            for name, total in total_train_metrics.items()
        }
        avg_train_loss = total_train_loss / num_train_batches

        # Store training metrics
        history.add('train_loss', avg_train_loss)
        history.add_many(avg_train_metrics, 'train_')

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0
        total_val_metrics: dict[str, float] = {}

        val_loop = tqdm(
            val_loader,
            desc=f'Epoch {epoch}/{epochs} [Val]',
            leave=False
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
                for name, value in batch_metrics.items():
                    total_val_metrics[name] = \
                        total_val_metrics.get(name, 0.0) + value
                total_val_loss += loss.item()

                # Update tqdm postfix
                val_loop.set_postfix(loss=loss.item())

        # Compute average by validation batch
        num_val_batches = len(val_loader)
        avg_val_metrics = {
            name: total / num_val_batches
            for name, total in total_val_metrics.items()
        }
        avg_val_loss = total_val_loss / num_val_batches

        history.add('val_loss', avg_val_loss)
        history.add_many(avg_val_metrics, 'val_')

        # Print epoch summary if verbose
        if verbose:
            train_metric_str = ' | '.join(
                [f'Train {name}: {value:.4f}'
                 for name, value in avg_train_metrics.items()]
            )
            val_metric_str = ' | '.join(
                [f'Val {name}: {value:.4f}'
                 for name, value in avg_val_metrics.items()]
            )
            print(f'Epoch {epoch:3d} | '
                  f'Train Loss: {avg_train_loss:.4f} | {train_metric_str} | '
                  f'Val Loss: {avg_val_loss:.4f} | {val_metric_str}')

        # Save checkpoint if interval
        if epoch % checkpoint_interval == 0:
            print('Saving checkpoint...', end='')
            torch.save(model.state_dict(),
                       path.join(checkpoint_path, f'{model_name}_checkpoint_{epoch}.model'))
            torch.save(optimizer.state_dict(),
                       path.join(checkpoint_path, f'{model_name}_checkpoint_{epoch}.optim'))
            history.dump(path.join(checkpoint_path,
                         f'{model_name}_checkpoint_{epoch}.metrics'))
            print('OK')

        history.next_epoch()

    return history.to_dict()


def test(model: nn.Module,
         dataloader: DataLoader,
         metrics: Callable[[torch.Tensor, torch.Tensor], dict[str, float]],
         device: torch.device = torch.device('cpu')) -> dict[str, float]:
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

    test_loop = tqdm(dataloader, desc='[Test]', leave=False)
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

    return {
        name: total / len(dataloader)
        for name, total in total_metrics.items()
    }
