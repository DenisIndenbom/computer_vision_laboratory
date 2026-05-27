import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ViT_B_16_Weights, vit_b_16

from methods import TrainArgs, register
from methods.common import VisDA2017Source, VisDA2017Validation
from utils.metrics import accuracy
from utils.trainer import train
from utils.transforms import base_transforms, train_source_transforms


@register('vitb16_baseline')
def vitb16_pretrain(args: TrainArgs):
    if not torch.cuda.is_available() and args['device'].startswith('cuda'):
        raise Exception('cuda is not available')

    # Load env vars
    imagenet_weights = bool(os.getenv('IMAGENET_WEIGHTS', 'false') == 'true')

    # Prepare arguments
    summary_writer = SummaryWriter(os.path.join(args['logs'], args['name']))
    checkpoint_path = os.path.join(args['checkpoint_path'], args['name'])
    device = torch.device(args['device'])

    # Load datasets
    train_dataset = VisDA2017Source(args['data'], transform=train_source_transforms, download=True)
    val_dataset = VisDA2017Validation(args['data'], transform=base_transforms, download=True)

    # Setup dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=args['workers'],
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args['workers'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Setup model and optimizer
    if not imagenet_weights:
        model = vit_b_16(num_classes=12)
    else:
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = nn.Linear(model.hidden_dim, 12)  # type: ignore

    optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'])
    loss = nn.CrossEntropyLoss()

    # Launch training
    train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss,
        accuracy,
        epochs=args['epochs'],
        start_epoch=args['start_epoch'],
        checkpoint_interval=args['checkpoint_interval'],
        checkpoint_path=checkpoint_path,
        device=device,
        seed=args['seed'],
        verbose=args['verbose'],
        summary_writer=summary_writer,
    )
