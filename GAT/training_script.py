# training_script.py
import torch
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from gat_model import GATEvolutionModel
from early_stopping import EarlyStopping
from outils import kendall_tau
import pandas as pd
import os

def train(model, optimizer, data, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    data.to(device)  # Move data to the appropriate device
    optimizer.zero_grad()
    out = model(data)
    out = out.view(-1)
    loss = criterion(out[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    return loss

def val(model, data, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    data.to(device)  # Move data to the appropriate device
    out = model(data)
    out = out.view(-1)
    loss = criterion(out[data.val_mask], data.y[data.val_mask].to(device))
    return loss

def test(model, data, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    data.to(device)  # Move data to the appropriate device
    out = model(data)
    out = out.view(-1)
    loss = criterion(out[data.test_mask], data.y[data.test_mask].to(device))
    sum_of_node_weights = torch.sum(data.y[data.test_mask].to(device))
    return loss, sum_of_node_weights, out

def train_val_test(data_list, hidden_channels, in_features, out_features, num_epochs, n_heads, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg_test_loss = 0
    avg_relative_mse = 0
    avg_correlation = 0
    avg_p_value = 0
    graphs_results = []

    for data in data_list:
        model = GATEvolutionModel(in_features=in_features, hidden_channels=hidden_channels, out_features=out_features, n_heads=n_heads).to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(patience=4)
        best_val_loss = float('inf')
        best_graph_result = None

        for epoch in tqdm(range(1, num_epochs + 1), desc=f"Training graph {data.graph_id}"):
            train_loss = train(model, optimizer, data, criterion)
            val_loss = val(model, data, criterion)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_graph_result = {
                    'graph_id': data.graph_id,
                    'train_loss': str(train_loss.item()).replace('.', ','),
                    'val_loss': str(val_loss.item()).replace('.', ','),
                    'epoch_stopped': epoch,
                    'test_loss': '',
                    'p_value': ''
                }

            if early_stopping.check_early_stop(val_loss):
                print(f"Arrêt anticipé à l'epoch {epoch} avec une perte de validation de {val_loss:.4f}")
                break

        test_loss, sum_of_weights, out = test(model, data, criterion)
        relative_mse = (test_loss / sum_of_weights)
        p_value, correlation = kendall_tau(data.y, out.detach().numpy())

        avg_test_loss += test_loss
        avg_relative_mse += float(relative_mse)
        avg_correlation += correlation
        avg_p_value += p_value

        best_graph_result['test_loss'] = str(test_loss.item()).replace('.', ',')
        best_graph_result['p_value'] = str(p_value.item()).replace('.', ',')

        graphs_results.append(best_graph_result)

        # Enregistrer les prédictions dans un fichier CSV
        predictions_df = pd.DataFrame({'Prediction': out.detach().numpy()})
        predictions_file_path = os.path.join("./predictions", f"{data.graph_id}_prediction.csv")
        predictions_df.to_csv(predictions_file_path, index=False)

    avg_test_loss /= len(data_list)
    avg_relative_mse /= len(data_list)
    avg_correlation /= len(data_list)
    avg_p_value /= len(data_list)

    print(f"Test loss: {avg_test_loss:.4f}, Relative MSE: {avg_relative_mse:.4f}, Correlation: {avg_correlation:.4f}, P-value: {avg_p_value:.4f}")

    return avg_test_loss, avg_relative_mse, avg_correlation, avg_p_value, model, graphs_results