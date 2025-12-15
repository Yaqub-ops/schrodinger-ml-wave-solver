import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from src.model import EigenNet
from datasets import multiFileDataset
import os

def train_model(dataset_path, model_path = None, num_epochs = 100, batch_size = 128):
    file_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".npz")]
    dataset = multiFileDataset(file_paths)

    X0, E0, PSI0 = dataset[0]
    input_dim = X0.shape[0]
    k = E0.shape[0]
    psi_dim = PSI0.shape[1]

    model = EigenNet(input_dim=input_dim, k=k, psi_dim=psi_dim)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model weights from {model_path}")
    else:
        print("Training a new model from scratch")

    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, E_batch, PSI_batch in train_loader:
            optimizer.zero_grad()
            pred_E, pred_PSI = model(X_batch)
            loss_E = criterion(pred_E, E_batch)
            loss_PSI = criterion(pred_PSI, PSI_batch)
            loss = loss_E + loss_PSI
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)  # accumulate weighted by batch size

        train_loss /= train_size
        loss_history.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, E_val, PSI_val in val_loader:
                pred_E, pred_PSI = model(X_val)
                loss_E = criterion(pred_E, E_val)
                loss_PSI = criterion(pred_PSI, PSI_val)
                val_loss += (loss_E + loss_PSI).item() * X_val.size(0)
        val_loss /= val_size
        val_loss_history.append(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "eigen_net_trained.pth")

    # Plot loss curves
    plt.plot(loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train EigenNet model")
    parser.add_argument("--model-path", type=str, default=None, help="Path to pre-trained model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (.npz) to run predictions on")
    parser.add_argument("--output-prefix", type=str, default="results", help="Prefix for saved files")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size for training")
    args = parser.parse_args()
    train_model(dataset_path=args.dataset, model_path=args.model_path, num_epochs=args.epochs, batch_size=args.batch_size)
