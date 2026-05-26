#!/usr/bin/env python3
from matplotlib import pyplot as plt
import json
import argparse
import sys


def plot_training_metrics(metrics_dict):
    """
    Plot training and validation metrics from a dictionary.

    Args:
        metrics_dict: Dictionary containing lists of metrics for each epoch
                      Expected keys: 'train_loss', 'train_acc1', 'train_acc5',
                                   'val_loss', 'val_acc1', 'val_acc5'
    """
    epochs = range(1, len(metrics_dict['train_loss']) + 1)

    # Create figure with subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Loss curves
    ax1.plot(epochs, metrics_dict['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, metrics_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy curves (Top-1 and Top-5)
    ax2.plot(epochs, metrics_dict['train_acc1'], 'b-', label='Train Top-1', linewidth=2)
    ax2.plot(epochs, metrics_dict['val_acc1'], 'r-', label='Val Top-1', linewidth=2)
    ax2.plot(epochs, metrics_dict['train_acc5'], 'b--', label='Train Top-5', linewidth=2, alpha=0.7)
    ax2.plot(epochs, metrics_dict['val_acc5'], 'r--', label='Val Top-5', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(
        description='Plot training metrics from a JSON file',
    )

    parser.add_argument('file', nargs='?', help='Path to the JSON file containing training metrics')

    parser.add_argument(
        '-f',
        '--file',
        dest='filename',
        help='Path to the JSON file containing training metrics (alternative to positional argument)',
    )

    parser.add_argument(
        '-s',
        '--save',
        metavar='PATH',
        help='Save the plot to file instead of displaying it (e.g., -s plot.png)',
    )

    parser.add_argument(
        '--no-show', action='store_true', help='Do not display the plot (useful with --save)'
    )

    parser.add_argument('--dpi', type=int, default=100, help='DPI for saved figure (default: 100)')

    args = parser.parse_args()

    # Determine the filename
    if args.filename:
        filename = args.filename
    elif args.file:
        filename = args.file
    else:
        parser.print_help()
        sys.exit(1)

    # Load and plot the metrics
    try:
        with open(filename, 'r') as file:
            metrics = json.load(file)

        plot_training_metrics(metrics)

        # Save the plot if requested
        if args.save:
            plt.savefig(args.save, dpi=args.dpi, bbox_inches='tight')
            print(f'Plot saved to {args.save}')

        # Show the plot if not suppressed
        if not args.no_show:
            plt.show()

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f'Error: Invalid JSON file - {e}', file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f'Error: Missing required metric key in JSON - {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
