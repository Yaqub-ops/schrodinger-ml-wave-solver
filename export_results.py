import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.model import EigenNet
from src.data_loader import load_dataset


def export_predictions(dataset_path, model_path, output_prefix="Results", batch_size=128):

    # Load dataset
    train_loader,val_loader, test_loader, train_size, val_size, test_size, input_dim, k , psi_dim  = load_dataset(dataset_path, batch_size=batch_size)

    # Initialize model
    model = EigenNet(input_dim=input_dim, k=k, psi_dim=psi_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Run predictions on the test dataset
    all_X = []
    all_Y_true = []
    all_pred_E = []
    all_pred_PSI = []

    with torch.no_grad():
        for X_batch, E_batch, PSI_batch in test_loader:
            pred_E, pred_PSI = model(X_batch)
            all_X.append(X_batch.numpy())
            all_Y_true.append(E_batch.numpy())
            all_pred_E.append(pred_E.numpy())
            all_pred_PSI.append(pred_PSI.numpy())

    # Concatenate batches
    all_X = np.vstack(all_X)
    all_Y_true = np.vstack(all_Y_true)
    all_pred_E = np.vstack(all_pred_E)
    all_pred_PSI = np.vstack(all_pred_PSI)

    # Save predictions to .npz
    np.savez(f"{output_prefix}_predictions.npz",
             X=all_X,
             Y_E_true=all_Y_true,
             Y_E_pred=all_pred_E,
             Y_PSI_pred=all_pred_PSI)
    print(f"Predictions saved to {output_prefix}_predictions.npz")

    # Plot first 5 predicted wavefunctions
    plt.figure(figsize=(8, 5))
    x = np.linspace(0,1,1001)
    for i in range(min(3 ,all_pred_PSI.shape[0])):
        for j in range(min(2, all_pred_E.shape[0])):
            plt.plot(x,all_pred_PSI[i][j], label=f"S{i+1}, E{j+1}")
    plt.title("Predicted wavefunctions")
    plt.xlabel("Spatial index")
    plt.ylabel("Ψ")
    plt.legend()
    plt.savefig(f"{output_prefix}_wavefunctions.png")
    plt.show()
    print(f"Wavefunction plot saved to {output_prefix}_wavefunctions.png")

    # Plot predicted vs true eigenvalues
    plt.figure(figsize=(6, 6))
    for i in range(all_Y_true.shape[1]):  # loop over k eigenvalues
        plt.scatter(all_Y_true[:, i], all_pred_E[:, i], label=f"Eigenvalue {i + 1}")
    plt.plot([all_Y_true.min(), all_Y_true.max()],
             [all_Y_true.min(), all_Y_true.max()], 'k--', label="y=x")
    plt.xlabel("True Eigenvalues")
    plt.ylabel("Predicted Eigenvalues")
    plt.title("Predicted vs True Eigenvalues")
    plt.legend()
    plt.savefig(f"{output_prefix}_eigenvalues.png")
    plt.show()
    print(f"Eigenvalue comparison plot saved to {output_prefix}_eigenvalues.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export predictions from trained EigenNet model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (.npz) to run predictions on")
    parser.add_argument("--output-prefix", type=str, default="results", help="Prefix for saved files")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size for training")

    args = parser.parse_args()
    export_predictions(dataset_path=args.dataset, model_path=args.model_path, output_prefix=args.output_prefix, batch_size=args.batch_size)

