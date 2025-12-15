import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


# Load dataset from file
def load_dataset(path, batch_size = 64, train_ratio = 0.75 ,val_ratio = 0.125,test_ratio = 0.125) :
    data = np.load(path)
    X = data['X']
    Y_E = data['Y_E']
    Y_PSI = data['Y_PSI']
    input_dim = X.shape[1]
    k = Y_E.shape[1]
    psi_dim = Y_PSI.shape[2]

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X).float()
    Y_E_tensor = torch.from_numpy(Y_E).float()
    Y_PSI_tensor = torch.from_numpy(Y_PSI).float()


# Create a dataset and dataloader
    dataset = TensorDataset(X_tensor, Y_E_tensor, Y_PSI_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    val_size = int(val_ratio * len(dataset))
    train_size = int(len(dataset)*train_ratio)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset,test_dataset = random_split(dataset, [train_size, val_size ,test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    M = X_tensor.shape[1]
    k = Y_E_tensor.shape[1]
    return train_loader,val_loader, test_loader, train_size, val_size, test_size, input_dim, k, psi_dim


