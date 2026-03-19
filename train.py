#!/usr/bin/env python3
import argparse
import sys

from pathlib import Path


def validate_args(args: argparse.Namespace):
    if not isinstance(args.name, str) or not args.name.strip():
        print('Error: --name must be a non‑empty string.')
        sys.exit(1)

    if not args.data.exists():
        print('Error: --data not found.')
        sys.exit(1)

    if not args.data.is_dir():
        print('Error: --data must be a directory.')
        sys.exit(1)

    if args.epochs <= 0:
        print('Error: --epochs must be a positive integer.')
        sys.exit(1)

    if args.start_epoch < 0:
        print('Error: --start_epoch cannot be negative.')
        sys.exit(1)
    if args.start_epoch > args.epochs:
        print('Error: --start_epoch cannot be greater than --epochs.')
        sys.exit(1)

    if args.batch_size <= 0:
        print('Error: --batch_size must be a positive integer.')
        sys.exit(1)

    if args.learning_rate <= 0:
        print('Error: --learning_rate must be positive.')
        sys.exit(1)

    if args.workers < 0:
        print('Error: --workers cannot be negative.')
        sys.exit(1)

    ckpt_path: Path = args.checkpoint_path
    ckpt_interval: int = args.checkpoint_interval

    if not ckpt_path.exists():
        print('Error: --checkpoint_path not found.')
        sys.exit(1)

    if not ckpt_path.is_dir():
        print('Error: --checkpoint_path must be a directory.')
        sys.exit(1)

    if ckpt_interval <= 0:
        print('Error: --checkpoint_interval must be a positive integer.')
        sys.exit(1)

    exp_dir = ckpt_path / args.name
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True)


def main():
    parser = argparse.ArgumentParser(
        description='Launch training for a specific method.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Basic arguments
    parser.add_argument(
        '--trainer', '-t',
        required=True,
        help='Name of the training method (must be registered in METHOD_REGISTRY).'
    )
    parser.add_argument(
        '--name', '-n',
        required=True,
        help='Run name – used for logs and checkpoint subdirectory.'
    )
    parser.add_argument(
        '--data', '-d',
        type=Path, default=Path('./data'),
        help='Path to the dataset directory.'
    )

    # Training hyperparameters
    train_group = parser.add_argument_group('Training hyperparameters')
    train_group.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs.'
    )
    train_group.add_argument(
        '--start_epoch', type=int, default=0,
        help='Epoch to resume training from (usually 0).'
    )
    train_group.add_argument(
        '--batch_size', type=int, default=128,
        help='Samples per batch.'
    )
    train_group.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help='Initial learning rate.'
    )
    train_group.add_argument(
        '--workers', type=int, default=4,
        help='Number of dataloader workers.'
    )
    train_group.add_argument(
        '--seed', type=int, default=42,
        help='Random seed.'
    )

    # Checkpointing & logging
    ckpt_group = parser.add_argument_group('Checkpointing')
    ckpt_group.add_argument(
        '--checkpoint_path', type=Path, default=Path('./checkpoints'),
        help='Directory where checkpoints are saved.'
    )
    ckpt_group.add_argument(
        '--checkpoint_interval', type=int, default=5,
        help='Save checkpoint every N epochs.'
    )

    # Hardware & verbosity
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument(
        '--device', type=str, default='cuda:0',
        help='Device to use (e.g. `cuda: 0`, `cpu`).'
    )
    misc_group.add_argument(
        '--verbose', action='store_true',
        help='Print detailed progress and metrics.'
    )

    args = parser.parse_args()

    validate_args(args)

    # Import module here, since torch is heavy
    from methods import METHOD_REGISTRY, TrainArgs

    if args.trainer not in METHOD_REGISTRY:
        print(
            f'Error: Trainer `{args.trainer}` not found. Available: {list(METHOD_REGISTRY.keys())}')
        sys.exit(1)

    train_args: TrainArgs = {
        'name': args.name,
        'data': str(args.data),
        'epochs': args.epochs,
        'start_epoch': args.start_epoch,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'workers': args.workers,
        'seed': args.seed,
        'checkpoint_path': str(args.checkpoint_path),
        'checkpoint_interval': args.checkpoint_interval,
        'verbose': args.verbose,
        'device': args.device,
    }

    print(train_args)

    # Launch training
    METHOD_REGISTRY[args.trainer](train_args)


if __name__ == '__main__':
    main()
