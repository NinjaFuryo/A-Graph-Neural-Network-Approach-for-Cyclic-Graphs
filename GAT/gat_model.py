#gat_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEvolutionModel(nn.Module):
    def __init__(self, in_features, hidden_channels, out_features, n_heads):
        super(GATEvolutionModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = GATConv(in_features, hidden_channels, heads=n_heads, edge_dim=1).to(self.device)
        self.conv2 = GATConv(hidden_channels * n_heads, out_features, heads=1, edge_dim=1).to(self.device)
        self.fc = nn.Linear(out_features, 1).to(self.device)
        self.model = nn.DataParallel(self).to(self.device)  # Utilisation de DataParallel

    def forward(self, data):
        # Vérifications des dimensions des données
        assert data.edge_attr.dim() == 1, "Edge attributes must be one-dimensional"
        assert data.edge_attr.size(0) == data.edge_index.size(1), "Mismatch between number of edges and edge attributes"

        # Transfert des tensors au bon appareil
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device).float()

        # Première couche GATConv avec dropout et activation ELU
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))

        # Deuxième couche GATConv suivie d'un dropout
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)

        # Application de la couche fully connected à la sortie de la dernière couche GATConv
        x = self.fc(x)

        return x





