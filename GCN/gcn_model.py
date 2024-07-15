#gat_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEvolutionModel(nn.Module):
    def __init__(self, in_features, hidden_channels, out_features):
        super(GCNEvolutionModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialisation des couches GCNConv

        self.conv1 = GCNConv(in_features, hidden_channels).to(self.device)
        self.conv2 = GCNConv(hidden_channels, out_features).to(self.device)

        # Couche fully connected qui suit la dernière couche GCNConv
        self.fc = nn.Linear(out_features, 1).to(self.device)

    def forward(self, data):
        x, edge_index, edge_weight = data.x.to(self.device), data.edge_index.to(self.device), data.edge_attr.to(
            self.device)

        # Première couche GCNConv avec dropout et activation ReLU
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))

        # Deuxième couche GCNConv suivie d'un dropout
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        # Application de la couche fully connected à la sortie de la dernière couche GCNConv
        x = self.fc(x)

        return x



