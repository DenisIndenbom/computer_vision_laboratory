import os
import torch

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.models import vit_b_16

from utils.dataset import BaseImageFolderDataset
from utils.transforms import base_transforms, train_transforms
from utils.metrics import accuracy
from utils.trainer import train, set_torch_seed

from methods import TrainArgs, register


class VisDA2017Train(BaseImageFolderDataset):
    URL = 'http://csr.bu.edu/ftp/visda17/clf/train.tar'
    ARCHIVE_NAME = 'train.tar'
    EXTRACTED_FOLDER = 'train'


class VisDA2017Validation(BaseImageFolderDataset):
    URL = ' http://csr.bu.edu/ftp/visda17/clf/validation.tar'
    ARCHIVE_NAME = 'validation.tar'
    EXTRACTED_FOLDER = 'validation'


@register('vitb16_pretrain')
def vitb16_pretrain(args: TrainArgs):
    set_torch_seed(args['seed'])

    # Load datasets
    train_dataset = VisDA2017Train(
        './data', transform=train_transforms, download=True
    )
    val_dataset = VisDA2017Validation(
        './data', transform=base_transforms, download=True
    )

    # Setup dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Setup model and optimizer
    model = vit_b_16(num_classes=12)
    optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'])
    loss = nn.CrossEntropyLoss()

    summary_writer = SummaryWriter(os.path.join('./logs', args['run_name']))

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
        checkpoint_path=args['checkpoint_path'],
        device=torch.device('cuda:0'),
        summary_writer=summary_writer
    )
