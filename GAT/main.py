# main.py

from training_script import train_val_test
from data_loader import load_and_split_data
import pandas as pd
import os
import torch


def main():
    # Définir les hyperparamètres
    learning_rate = 0.0002
    num_epochs = 400
    in_features = 1  # Taille de la variable d'entrée
    hidden_channels = 64
    out_features = 1  # Taille de la sortie
    n_heads = 8

    # Charger et préparer les données
    data_list = load_and_split_data()  # Assurer que le chargement prend en compte le périphérique

    # Entraînement, validation, et test sur les ensembles de données complets.
    test_loss, relative_mse, correlation, p_value, trained_model, graphs_results = train_val_test(
        data_list, hidden_channels, in_features, out_features, num_epochs, n_heads, learning_rate)

    print(f"Test loss: {test_loss:.4f}, Relative MSE: {relative_mse}, Correlation: {correlation}, P-value: {p_value}")

    # Enregistrer graphs_results dans un DataFrame
    graphs_results_df = pd.DataFrame(graphs_results)

    # Enregistrer le DataFrame dans un fichier CSV
    results_folder = "./results/"
    os.makedirs(results_folder, exist_ok=True)  # Créer le dossier "results" s'il n'existe pas
    graphs_results_file_path = os.path.join(results_folder, "graphs_results.csv")
    graphs_results_df.to_csv(graphs_results_file_path, index=False)


if __name__ == "__main__":
    main()
