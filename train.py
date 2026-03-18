import argparse


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group('Training options')
    group.add_argument('--method', type=str,
                       help='Method name (module in methods/)')
    group.add_argument('--name', type=str,
                       help='Run name in logs/')
    group.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    group.add_argument('--start_epoch', type=int, default=0,
                       help='Epoch to start/resume training from')
    group.add_argument('--batch_size', type=int, default=128,
                       help='Number of samples per batch during training')
    group.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate for optimizer')
    group.add_argument('--workers', type=int, default=4,
                       help='Number of dataloader worker processes')
    group.add_argument('--seed', type=int, default=42,
                       help='Random seed for numpy/torch random number generators')
    group.add_argument('--checkpoint_path', type=str, default='./checkpoints',
                       help='Directory path where model checkpoints will be saved')
    group.add_argument('--checkpoint_interval', type=int, default=5,
                       help='Number of epochs between saving model checkpoints')
    group.add_argument('--verbose', type=bool, default=True,
                       help='Print detailed training progress and metrics (True/False)')

    args = parser.parse_args()

    from methods import METHOD_REGISTRY, TrainArgs

    train_args: TrainArgs = {
        'run_name': args.name,
        'epochs': args.epochs,
        'start_epoch': args.start_epoch,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'workers': args.workers,
        'seed': args.seed,
        'checkpoint_interval': args.checkpoint_interval,
        'checkpoint_path': args.checkpoint_path,
        'verbose': args.verbose
    }

    METHOD_REGISTRY[args.method](train_args)


if __name__ == '__main__':
    main()
