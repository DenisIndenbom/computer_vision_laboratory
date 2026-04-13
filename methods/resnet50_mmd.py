import os
import torch

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.models import resnet50

from typing import cast

from utils.dataset import BaseImageFolderDataset, DomainDataset
from utils.sampler import DomainBatchSampler
from utils.transforms import base_transforms, train_source_transforms, train_target_transforms
from utils.criterion import MMD
from utils.metrics import bundle, metrics_with_mask, accuracy, mmd
from utils.trainer import train
from utils.typing import TrainHookF

from methods import TrainArgs, register


class VisDA2017Source(BaseImageFolderDataset):
    URL = 'http://csr.bu.edu/ftp/visda17/clf/train.tar'
    ARCHIVE_NAME = 'train.tar'
    EXTRACTED_FOLDER = 'train'


class VisDA2017Target(BaseImageFolderDataset):
    URL = 'http://csr.bu.edu/ftp/visda17/clf/test.tar'
    ARCHIVE_NAME = 'test.tar'
    EXTRACTED_FOLDER = 'test'


class VisDA2017Validation(BaseImageFolderDataset):
    URL = 'http://csr.bu.edu/ftp/visda17/clf/validation.tar'
    ARCHIVE_NAME = 'validation.tar'
    EXTRACTED_FOLDER = 'validation'


class Criterion(nn.Module):
    def __init__(self, model, lambda_mmd: float = 0.5):
        super().__init__()
        self.model = model
        self.lambda_mmd = lambda_mmd

        self.ce = nn.CrossEntropyLoss()
        self.mmd = MMD()

    def forward(self, pred: torch.Tensor, y: torch.Tensor):
        mask = y != -1

        cls_loss = self.ce(pred[mask], y[mask])

        mmd = torch.tensor(0, device=y.device)

        if (~mask).sum() > 0:
            feat_l1 = self.model._feat_l1
            feat_l3 = self.model._feat_l3
            feat_fl = self.model._features

            mmd_1 = self.mmd(feat_l1[mask], feat_l1[~mask])
            mmd_2 = self.mmd(feat_l3[mask], feat_l3[~mask])
            mmd_3 = self.mmd(feat_fl[mask], feat_fl[~mask])

            mmd_s = torch.stack([mmd_1, mmd_2, mmd_3])
            weights = torch.softmax(mmd_s.detach(), dim=0)
            mmd = (weights * mmd_s).sum()

        return cls_loss + self.lambda_mmd * mmd


def build_hook(
    start_lambda: float, end_lambda: float, start_epoch: int, end_epoch: int
) -> TrainHookF:
    def hook(epoch: int, model: nn.Module, optim: optim.Optimizer, criterion: nn.Module) -> None:
        t = min(max((epoch - start_epoch) / (end_epoch - start_epoch), 0.0), 1.0)
        new_lambda = start_lambda + t * (end_lambda - start_lambda)

        cast(Criterion, criterion).lambda_mmd = new_lambda
        print(f'New lambda: {new_lambda}')

    return hook


@register('resnet50_mmd')
def resnet50_mmd(args: TrainArgs):
    if not torch.cuda.is_available() and args['device'].startswith('cuda'):
        raise Exception('cuda is not available')

    # Load env vars
    mmd_lambda_start = float(os.getenv('MMD_LAMBDA_START', 0.5))
    mmd_lambda_end = float(os.getenv('MMD_LAMBDA_END', 0.5))
    mmd_start_epoch = int(os.getenv('MMD_START_EPOCH', 0))
    mmd_ramp_epochs = int(os.getenv('MMD_RAMP_EPOCHS', 0))

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
    model = resnet50(num_classes=12)
    model.layer1.register_forward_hook(lambda m, i, o: setattr(model, '_feat_l1', o.mean([2, 3])))
    model.layer3.register_forward_hook(lambda m, i, o: setattr(model, '_feat_l3', o.mean([2, 3])))
    model.avgpool.register_forward_hook(
        lambda m, i, o: setattr(model, '_features', torch.flatten(o, 1))
    )

    # Setup optimizer, loss and metrics
    optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'])
    loss = Criterion(model, lambda_mmd=mmd_lambda_start)
    metrics = bundle([metrics_with_mask(accuracy), mmd])

    # Setup hook for updating lambda
    hook = None
    if mmd_ramp_epochs > 0:
        hook = build_hook(
            mmd_lambda_start, mmd_lambda_end, mmd_start_epoch, mmd_start_epoch + mmd_ramp_epochs
        )

    # Prepare arguments
    summary_writer = SummaryWriter(os.path.join(args['logs'], args['name']))
    checkpoint_path = os.path.join(args['checkpoint_path'], args['name'])
    device = torch.device(args['device'])

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
        seed=args['seed'],
        verbose=args['verbose'],
        summary_writer=summary_writer,
        post_epoch_hook=hook,
    )
