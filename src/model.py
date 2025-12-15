import torch
from torch import nn
import numpy as np

class EigenNet(nn.Module):
    def __init__(self, input_dim, k, psi_dim):
        super().__init__()

        self.k = k
        self.psi_dim = psi_dim

        # Shared feature extractor (unchanged)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Eigenvalue head now outputs k eigenvalues
        self.E_head = nn.Linear(128, k)

        # Psi head outputs k * psi_dim (flatten) that we reshape later
        self.psi_head = nn.Linear(128, k * psi_dim)

    def forward(self, x):
        h = self.shared(x)

        E = self.E_head(h)  # shape = (batch, k)

        psi = self.psi_head(h)          # shape = (batch, k * psi_dim)
        psi = psi.view(-1, self.k, self.psi_dim)  # → (batch, k, psi_dim)

        return E, psi

