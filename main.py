print("DEBUG: running main.py")
import sys
print("DEBUG: sys.argv =", sys.argv)

import argparse
import train
from export_results import export_predictions
import sys, os
print("DEBUG: main.py loaded")
print("DEBUG: sys.argv =", sys.argv)
print("DEBUG: current working directory =", os.getcwd())
print("DEBUG: main.py absolute path =", os.path.abspath(__file__))


def main():
    print("DEBUG: inside main()")

    parser = argparse.ArgumentParser(description="EigenNet Project Main Entry")

    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model or continue training")
    train_parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (.npz)")
    train_parser.add_argument("--model-path", type=str, default=None, help="Optional path to pre-trained model")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size for training")


    # Export command
    export_parser = subparsers.add_parser("export", help="Export predictions from trained model")
    export_parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (.npz) to run predictions on")
    export_parser.add_argument("--model-path", type=str, required=True, help="Path to trained model weights")
    export_parser.add_argument("--output-prefix", type=str, default="results", help="Prefix for saved files")
    export_parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size for training")


    args = parser.parse_args()

    if args.command == "train":

        train.train_model(dataset_path=args.dataset, model_path=args.model_path, num_epochs=args.epochs, batch_size = args.batch_size)

    elif args.command == "export":
        export_predictions(dataset_path=args.dataset, model_path=args.model_path, output_prefix=args.output_prefix, batch_size = args.batch_size)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
