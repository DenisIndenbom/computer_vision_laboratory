import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet50_Weights

from methods import TrainArgs, register
from utils.dataset import DomainDataset
from utils.gradient import freeze_rand_params
from utils.metrics import bundle
from utils.models import ResNetDANN
from utils.sampler import DomainBatchSampler
from utils.trainer import train
from utils.transforms import base_transforms, train_source_transforms, train_target_transforms

from .common import VisDA2017Source, VisDA2017Target, VisDA2017Validation


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()
        self.be = nn.BCEWithLogitsLoss()

    def forward(self, pred: tuple[torch.Tensor, torch.Tensor], y: torch.Tensor):
        class_label, domain_label = pred
        mask = y != -1

        class_loss = self.ce(class_label[mask], y[mask])

        if mask.all():
            domain_loss = self.ce(domain_label, mask)
        else:
            domain_loss = torch.tensor(0, device=y.device)

        return class_loss + domain_loss


@register('resnet50_dann')
def resnet50_dann(args: TrainArgs):
    if not torch.cuda.is_available() and args['device'].startswith('cuda'):
        raise Exception('cuda is not available')

    # Load env vars
    imagenet_weights = bool(os.getenv('IMAGENET_WEIGHTS', 'false') == 'true')
    freeze_ratio = float(os.getenv('FREEZE_RATIO', 0.0))
    dann_lambda = float(os.getenv('DANN_LAMBDA', 1.0))

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
    model = ResNetDANN(
        num_classes=12,
        lambda_=dann_lambda,
        weights=ResNet50_Weights.IMAGENET1K_V2 if imagenet_weights else None,
    )
    model.to(device)

    if freeze_ratio > 0.0:
        freeze_rand_params(
            model.named_parameters(), freeze_ratio, device, seed, ['fc.weight', 'fc.bias']
        )

    # Setup optimizer, loss and metrics
    optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'])
    loss = Criterion()
    # TODO: Add metric for domain classification
    metrics = bundle([])

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
