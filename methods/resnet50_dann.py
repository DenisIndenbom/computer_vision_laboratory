import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet50_Weights

from methods import TrainArgs, register
from utils.dataset import DomainDataset, split_dataset
from utils.gradient import freeze_params
from utils.metrics import (
    accuracy,
    binary_accuracy,
    bundle,
    with_mask,
    with_prefix,
    with_slice,
    with_transform,
    apply_if,
)
from utils.models import ResNetDANN
from utils.sampler import DomainBatchSampler
from utils.trainer import train
from utils.transforms import base_transforms, train_source_transforms, train_target_transforms

from .common import VisDA2017Source, VisDA2017Target, VisDA2017Validation


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

        self.class_loss = nn.CrossEntropyLoss()
        self.domain_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred: tuple[torch.Tensor, torch.Tensor], y: torch.Tensor):
        class_logits, domain_logits = pred
        mask = y != -1

        class_loss = torch.tensor(0.0, device=y.device)
        if mask.any():
            class_loss = self.class_loss(class_logits[mask], y[mask])

        domain_loss = torch.tensor(0.0, device=y.device)
        if (~mask).any():
            domain_target = (~mask).float().unsqueeze(1)
            domain_loss = self.domain_loss(domain_logits, domain_target)

        return class_loss + domain_loss


@register('resnet50_dann')
def resnet50_dann(args: TrainArgs):
    if not torch.cuda.is_available() and args['device'].startswith('cuda'):
        raise Exception('cuda is not available')

    # Load env vars
    imagenet_weights = bool(os.getenv('IMAGENET_WEIGHTS', 'false') == 'true')
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

    _, source_dataset = split_dataset(source_dataset, 0.1)
    _, target_dataset = split_dataset(target_dataset, 0.1)
    _, val_dataset = split_dataset(val_dataset, 0.1)

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
        model = ResNetDANN(num_classes=12, lambda_=dann_lambda)
    else:
        model = ResNetDANN(
            num_classes=12, lambda_=dann_lambda, weights=ResNet50_Weights.IMAGENET1K_V2
        )

        # Freeze params to prevent overfitting.
        to_exclude = [
            name
            for name, _ in model.named_parameters()
            if name.startswith(
                ('features.layer3', 'features.layer4', 'classifier', 'domain_classifier')
            )
        ]
        freeze_params(model.named_parameters(), exclude=to_exclude)
    model.to(device)

    # Setup optimizer, loss and metrics
    if imagenet_weights:
        param_groups = [
            {'params': model.classifier.parameters(), 'lr': args['learning_rate']},
            {'params': model.domain_classifier.parameters(), 'lr': args['learning_rate']},
            {'params': model.features.layer4.parameters(), 'lr': args['learning_rate'] / 10},  # type: ignore
            {'params': model.features.layer3.parameters(), 'lr': args['learning_rate'] / 10},  # type: ignore
        ]
    else:
        param_groups = [{'params': model.parameters(), 'lr': args['learning_rate']}]

    optimizer = optim.AdamW(param_groups)
    loss = Criterion()
    metrics = bundle(
        [
            with_slice(with_mask(accuracy), 0),
            apply_if(
                with_prefix(
                    with_slice(
                        with_transform(
                            binary_accuracy,
                            lambda pred, target: (pred, target == -1),
                        ),
                        1,
                    ),
                    'domain_',
                ),
                lambda pred, target: bool((target == -1).any().item()),
            ),
        ]
    )

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
