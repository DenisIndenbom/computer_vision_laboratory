import argparse

from methods import METHOD_REGISTRY, TrainArgs


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--method', type=str,
                       help='Method name (module in methods/)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Epoch to start/resume training from')
    parser.add_argument('--checkpoint_interval', type=int, default=5,
                        help='Number of epochs between saving model checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints',
                        help='Directory path where model checkpoints will be saved')
    parser.add_argument('--verbose', type=bool, default=True,
                        help='Print detailed training progress and metrics (True/False)')

    args = parser.parse_args()

    train_args: TrainArgs = {
        'epochs': args.epochs,
        'start_epoch': args.start_epoch,
        'checkpoint_interval': args.checkpoint_interval,
        'checkpoint_path': args.checkpoint_path,
        'verbose': args.verbose
    }

    METHOD_REGISTRY[args.method](train_args)


if __name__ == '__main__':
    main()
