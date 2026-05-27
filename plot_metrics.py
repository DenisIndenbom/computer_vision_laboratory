#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import argparse
import sys


def plot_training_metrics(metrics_dict: dict[str, list[int | float]]):
    """
    Plot training and validation metrics from a dictionary.
    Training and validation are drawn in separate axes for clarity.

    Args:
        metrics_dict: Dictionary containing lists of metrics for each epoch
                      Expected keys: 'train_loss', 'train_acc1', 'train_acc5',
                                   'val_loss', 'val_acc1', 'val_acc5'
    """
    epochs = range(1, len(metrics_dict['train_loss']) + 1)

    additional_metrics_keys = ['train_mmd', 'train_coral', 'train_domain_acc']
    additional_metric = next((key for key in additional_metrics_keys if key in metrics_dict), None)

    # Define figure size and grid layout
    if additional_metric:
        # 3 rows, 2 columns: top 2 rows for loss/acc, bottom row for additional metric
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.7])
        ax_train_loss = fig.add_subplot(gs[0, 0])
        ax_val_loss = fig.add_subplot(gs[1, 0])
        ax_train_acc = fig.add_subplot(gs[0, 1])
        ax_val_acc = fig.add_subplot(gs[1, 1])
        ax_additional = fig.add_subplot(gs[2, :])  # span both columns
    else:
        fig, ((ax_train_loss, ax_train_acc), (ax_val_loss, ax_val_acc)) = plt.subplots(
            2, 2, figsize=(12, 10)
        )
        ax_additional = None

    # Plot training and validation loss
    ax_train_loss.plot(epochs, metrics_dict['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax_train_loss.set_xlabel('Epochs')
    ax_train_loss.set_ylabel('Loss')
    ax_train_loss.set_title('Training Loss')
    ax_train_loss.legend()
    ax_train_loss.grid(True, alpha=0.3)

    ax_val_loss.plot(epochs, metrics_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax_val_loss.set_xlabel('Epochs')
    ax_val_loss.set_ylabel('Loss')
    ax_val_loss.set_title('Validation Loss')
    ax_val_loss.legend()
    ax_val_loss.grid(True, alpha=0.3)

    # Plot training and validation accuracy
    ax_train_acc.plot(epochs, metrics_dict['train_acc1'], 'b-', label='Top-1', linewidth=2)
    ax_train_acc.plot(
        epochs, metrics_dict['train_acc5'], 'b--', label='Top-5', linewidth=2, alpha=0.7
    )
    ax_train_acc.set_xlabel('Epochs')
    ax_train_acc.set_ylabel('Accuracy (%)')
    ax_train_acc.set_title('Training Accuracy')
    ax_train_acc.legend()
    ax_train_acc.grid(True, alpha=0.3)

    ax_val_acc.plot(epochs, metrics_dict['val_acc1'], 'r-', label='Top-1', linewidth=2)
    ax_val_acc.plot(epochs, metrics_dict['val_acc5'], 'r--', label='Top-5', linewidth=2, alpha=0.7)
    ax_val_acc.set_xlabel('Epochs')
    ax_val_acc.set_ylabel('Accuracy (%)')
    ax_val_acc.set_title('Validation Accuracy')
    ax_val_acc.legend()
    ax_val_acc.grid(True, alpha=0.3)

    # Additional metric plot if provided
    if additional_metric is not None and ax_additional is not None:
        label = additional_metric.replace('train_', '').upper()
        ax_additional.plot(epochs, metrics_dict[additional_metric], 'g-', label=label, linewidth=2)
        ax_additional.set_xlabel('Epochs')
        ax_additional.set_ylabel('Metric Value')
        ax_additional.set_title(f'Training {label}')
        ax_additional.legend()
        ax_additional.grid(True, alpha=0.3)

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

    parser.add_argument('--dpi', type=int, default=150, help='DPI for saved figure (default: 150)')

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
