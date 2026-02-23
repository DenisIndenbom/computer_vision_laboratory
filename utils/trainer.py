import torch
import torch.nn as nn

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import accuracy


def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optimizer: Optimizer,
          criterion: nn.Module,
          epochs: int,
          device: torch.device = torch.device('cpu'),
          verbose: bool = True) -> dict[str, list[float]]:
    """
    Train the model for a given number of epochs, evaluating on the validation set after each epoch.

    Args:
        model: Neural network model.
        train_loader: DataLoader supplying training batches.
        val_loader: DataLoader supplying validation batches.
        optimizer: Optimizer for updating model weights.
        criterion: Loss function.
        epochs: Number of training epochs.
        device: Device on which to perform computation.
        verbose: If True, print per-epoch metrics.

    Returns:
        Dictionary containing lists of metrics for each epoch:
        - 'train_loss', 'train_acc1', 'train_acc5'
        - 'val_loss', 'val_acc1', 'val_acc5'
    """
    # Move model to device
    model.to(device)

    # History storage
    history = {
        'train_loss': [],
        'train_acc1': [],
        'train_acc5': [],
        'val_loss': [],
        'val_acc1': [],
        'val_acc5': []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        acc1 = 0.0
        acc5 = 0.0

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
            total_loss += loss.item()
            acc = accuracy(pred, y_batch)
            acc1 += acc[0].item() if torch.is_tensor(acc[0]) else float(acc[0])
            acc5 += acc[1].item() if torch.is_tensor(acc[1]) else float(acc[1])

            # Update tqdm postfix
            train_loop.set_postfix(loss=loss.item(),
                                   acc1=acc[0].item() if torch.is_tensor(acc[0]) else acc[0])

        num_train_batches = len(train_loader)
        avg_train_loss = total_loss / num_train_batches
        avg_train_acc1 = acc1 / num_train_batches
        avg_train_acc5 = acc5 / num_train_batches

        # Store training metrics
        history['train_loss'].append(avg_train_loss)
        history['train_acc1'].append(avg_train_acc1)
        history['train_acc5'].append(avg_train_acc5)

        avg_val = validate(model, val_loader, criterion, device, epoch, epochs)

        # Store validation metrics
        history['val_loss'].append(avg_val[0])
        history['val_acc1'].append(avg_val[1])
        history['val_acc5'].append(avg_val[2])

        # Print epoch summary if verbose
        if verbose:
            print(f'Epoch {epoch:3d} | '
                  f'Train Loss: {avg_train_loss:.4f} | Train Acc1: {avg_train_acc1:.2f}% | Train Acc5: {avg_train_acc5:.2f}% | '
                  f'Val Loss: {avg_val[0]:.4f} | Val Acc1: {avg_val[1]:.2f}% | Val Acc5: {avg_val[2]:.2f}%')

    return history


def validate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device = torch.device('cpu'),
             epoch: int = 1,
             epochs: int = 1) -> tuple[float, float, float]:
    """
    Evaluate the model on a validation set.

    Args:
        model: Neural network model.
        dataloader: DataLoader supplying validation batches.
        criterion: Loss function.
        device: Device on which to perform computation.
        epoch: Current epoch.
        epochs: Total epochs.

    Returns:
        Tuple of (average loss, top‑1 accuracy, top‑5 accuracy) on the validation set.
    """
    model.eval()
    total_loss = 0.0
    acc1 = 0.0
    acc5 = 0.0

    val_loop = tqdm(
        dataloader, desc=f'Epoch {epoch}/{epochs} [Val]', leave=False)
    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader, desc='Validation'):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            # Update metrics
            total_loss += loss.item()
            acc = accuracy(pred, y_batch)
            acc1 += float(acc[0]) if not torch.is_tensor(acc[0]
                                                         ) else acc[0].item()
            acc5 += float(acc[1]) if not torch.is_tensor(acc[1]
                                                         ) else acc[1].item()

            val_loop.set_postfix(loss=loss.item(),
                                 acc1=acc[0].item() if torch.is_tensor(acc[0]) else acc[0])

    # Average over number of batches
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_acc1 = acc1 / num_batches
    avg_acc5 = acc5 / num_batches

    return avg_loss, avg_acc1, avg_acc5
