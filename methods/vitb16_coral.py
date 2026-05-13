import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ViT_B_16_Weights, vit_b_16

from methods import TrainArgs, register
from utils.criterion import Coral
from utils.dataset import DomainDataset
from utils.gradient import freeze_params
from utils.metrics import accuracy, bundle, distance_metric, with_mask
from utils.sampler import DomainBatchSampler
from utils.trainer import train
from utils.transforms import base_transforms, train_source_transforms, train_target_transforms
from utils.typing import MetricF

from .common import VisDA2017Source, VisDA2017Target, VisDA2017Validation


class Criterion(nn.Module):
    def __init__(self, model, lambda_coral: float = 0.5):
        super().__init__()
        self.model = model
        self.lambda_coral = lambda_coral

        self.ce = nn.CrossEntropyLoss()
        self.coral = Coral()

    def forward(self, pred: torch.Tensor, y: torch.Tensor):
        mask = y != -1

        cls_loss = self.ce(pred[mask], y[mask])

        if (~mask).sum() > 0:
            feat_l8 = self.model.feat_l8_
            feat_fl = self.model.feat_fl_

            coral_1 = self.coral(feat_l8[mask], feat_l8[~mask])
            coral_2 = self.coral(feat_fl[mask], feat_fl[~mask])

            coral = (coral_1 + coral_2) / 2.0
        else:
            coral = torch.tensor(0, device=y.device)

        return cls_loss + self.lambda_coral * coral


def coral_fl(model) -> MetricF:
    coral = distance_metric(Coral(), 'coral')

    def metric(_: torch.Tensor, target: torch.Tensor) -> dict[str, int | float]:
        feat_fl = model.feat_fl_

        return coral(feat_fl, target)

    return metric


@register('vitb16_coral')
def vitb16_mmd(args: TrainArgs):
    if not torch.cuda.is_available() and args['device'].startswith('cuda'):
        raise Exception('cuda is not available')

    # Load env vars
    imagenet_weights = bool(os.getenv('IMAGENET_WEIGHTS', 'false') == 'true')
    coral_lambda_start = float(os.getenv('CORAL_LAMBDA_START', 0.5))

    # Prepare arguments
    summary_writer = SummaryWriter(os.path.join(args['logs'], args['name']))
    checkpoint_path = os.path.join(args['checkpoint_path'], args['name'])
    device = torch.device(args['device'])
    seed = args['seed']

    # Load datasets
    source_dataset = VisDA2017Source(args['data'], transform=train_source_transforms, download=True)
    target_dataset = VisDA2017Target(args['data'], transform=train_target_transforms, download=True)
    val_dataset = VisDA2017Validation(args['data'], transform=base_transforms, download=True)

    train_dataset = DomainDataset(source_dataset, target_dataset)

    train_sampler = DomainBatchSampler(
        source_size=len(source_dataset),
        target_size=len(target_dataset),
        batch_size=args['batch_size'],
    )

    # Setup dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        num_workers=args['workers'],
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args['workers'],
        pin_memory=True,
    )

    # Setup model
    if not imagenet_weights:
        model = vit_b_16(num_classes=12)
    else:
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = nn.Linear(model.hidden_dim, 12)  # type: ignore

        # Freeze parameters to prevent overfitting
        to_exclude = [
            name
            for name, _ in model.named_parameters()
            if name.startswith(
                (
                    'encoder.layers.8',
                    'encoder.layers.9',
                    'encoder.layers.10',
                    'encoder.layers.11',
                    'heads',
                )
            )
        ]
        freeze_params(model.named_parameters(), exclude=to_exclude)

    model.encoder.layers[8].register_forward_hook(
        lambda m, i, o: setattr(model, 'feat_l8_', o[:, 0, :])
    )
    model.encoder.ln.register_forward_hook(lambda m, i, o: setattr(model, 'feat_fl_', o[:, 0, :]))
    model.to(device)

    # Setup optimizer, loss and metrics
    if imagenet_weights:
        param_groups = [
            {'params': model.heads.parameters(), 'lr': args['learning_rate']},
            {'params': model.encoder.layers[8].parameters(), 'lr': args['learning_rate'] / 10},
            {'params': model.encoder.layers[9].parameters(), 'lr': args['learning_rate'] / 10},
            {'params': model.encoder.layers[10].parameters(), 'lr': args['learning_rate'] / 10},
            {'params': model.encoder.layers[11].parameters(), 'lr': args['learning_rate'] / 10},
        ]
    else:
        param_groups = [{'params': model.parameters(), 'lr': args['learning_rate']}]

    optimizer = optim.AdamW(param_groups)
    loss = Criterion(model, lambda_coral=coral_lambda_start)
    metrics = bundle([with_mask(accuracy), coral_fl(model)])

    # Launch training
    train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss,
        metrics,
        epochs=args['epochs'],
        start_epoch=args['start_epoch'],
        checkpoint_interval=args['checkpoint_interval'],
        checkpoint_path=checkpoint_path,
        device=device,
        seed=seed,
        verbose=args['verbose'],
        summary_writer=summary_writer,
    )
