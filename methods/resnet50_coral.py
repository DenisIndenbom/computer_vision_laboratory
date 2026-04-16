import os
import torch

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.models import resnet50, ResNet50_Weights

from typing import cast

from utils.dataset import BaseImageFolderDataset, DomainDataset
from utils.sampler import DomainBatchSampler
from utils.transforms import base_transforms, train_source_transforms, train_target_transforms
from utils.criterion import Coral
from utils.gradient import freeze_params
from utils.metrics import bundle, metrics_with_mask, accuracy, coral
from utils.trainer import train
from utils.typing import TrainHookF, MetricF

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
    def __init__(self, model, lambda_coral: float = 0.5):
        super().__init__()
        self.model = model
        self.lambda_coral = lambda_coral

        self.ce = nn.CrossEntropyLoss()
        self.coral = Coral()

    def forward(self, pred: torch.Tensor, y: torch.Tensor):
        mask = y != -1

        cls_loss = self.ce(pred[mask], y[mask])

        coral = torch.tensor(0, device=y.device)

        if (~mask).sum() > 0:
            feat_l1 = self.model._feat_l1
            feat_l3 = self.model._feat_l3
            feat_fl = self.model._features

            coral_1 = self.coral(feat_l1[mask], feat_l1[~mask])
            coral_2 = self.coral(feat_l3[mask], feat_l3[~mask])
            coral_3 = self.coral(feat_fl[mask], feat_fl[~mask])

            coral = (coral_1 + coral_2 + coral_3) / 3.0

        return cls_loss + self.lambda_coral * coral


def build_hook(
    start_lambda: float, end_lambda: float, start_epoch: int, end_epoch: int
) -> TrainHookF:
    def hook(epoch: int, model: nn.Module, optim: optim.Optimizer, criterion: nn.Module) -> None:
        t = min(max((epoch - start_epoch) / (end_epoch - start_epoch), 0.0), 1.0)
        new_lambda = start_lambda + t * (end_lambda - start_lambda)

        cast(Criterion, criterion).lambda_coral = new_lambda
        print(f'New lambda: {new_lambda}')

    return hook


def coral_fl(model) -> MetricF:
    def metric(_: torch.Tensor, target: torch.Tensor) -> dict[str, int | float]:
        feat_fl = model._features

        return coral(feat_fl, target)

    return metric


@register('resnet50_coral')
def resnet50_coral(args: TrainArgs):
    if not torch.cuda.is_available() and args['device'].startswith('cuda'):
        raise Exception('cuda is not available')

    # Load env vars
    imagenet_weights = bool(os.getenv('IMAGENET_WEIGHTS', 'false') == 'true')
    freeze_ratio = float(os.getenv('FREEZE_RATIO', 0.0))
    coral_lambda_start = float(os.getenv('CORAL_LAMBDA_START', 0.1))
    coral_lambda_end = float(os.getenv('CORAL_LAMBDA_END', 0.1))
    coral_start_epoch = int(os.getenv('CORAL_START_EPOCH', 0))
    coral_ramp_epochs = int(os.getenv('CORAL_RAMP_EPOCHS', 0))

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
        model = resnet50(num_classes=12)
    else:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(512 * 4, 12)

    model.layer1.register_forward_hook(lambda m, i, o: setattr(model, '_feat_l1', o.mean([2, 3])))
    model.layer3.register_forward_hook(lambda m, i, o: setattr(model, '_feat_l3', o.mean([2, 3])))
    model.avgpool.register_forward_hook(
        lambda m, i, o: setattr(model, '_features', torch.flatten(o, 1))
    )
    model.to(device)

    if freeze_ratio > 0.0:
        freeze_params(
            model.named_parameters(), freeze_ratio, device, seed, ['fc.weight', 'fc.bias']
        )

    # Setup optimizer, loss and metrics
    optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'])
    loss = Criterion(model, lambda_coral=coral_lambda_start)
    metrics = bundle([metrics_with_mask(accuracy), coral_fl(model)])

    # Setup hook for updating lambda
    hook = None
    if coral_ramp_epochs > 0:
        hook = build_hook(
            coral_lambda_start,
            coral_lambda_end,
            coral_start_epoch,
            coral_start_epoch + coral_ramp_epochs,
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
        post_epoch_hook=hook,
    )
