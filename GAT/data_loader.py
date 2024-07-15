# data_loader.py

import os
import pandas as pd
import torch
from torch_geometric.data import Data
import torch_geometric.utils as utils
from scipy.sparse import coo_matrix

def from_file_to_matrix(graph_id):
    """
    Charge les données à partir des fichiers spécifiés pour un graph donné.

    Parameters:
    - graph_id (str): Identifiant du graph (par exemple, 'bag_100_49').

    Returns:
    - features_tensor (torch.Tensor): La matrice de feature.
    - adjacency_matrix (torch.Tensor): La matrice d'adjacence.
    """
    file_path_arguments = os.path.join("arguments", f"{graph_id}_arguments.csv")
    file_path_entailments = os.path.join("entailments", f"{graph_id}_entailment.csv")
    #file_path_arguments = os.path.join("/content/drive/My Drive/arguments", f"{graph_id}_arguments.csv")
    #file_path_entailments = os.path.join("/content/drive/My Drive/entailments", f"{graph_id}_entailment.csv")k

    # Chargement des données d'arguments
    if os.path.exists(file_path_arguments):
        data = pd.read_csv(file_path_arguments)

        # Diviser la colonne 'ArgName' en deux parties : la partie alphabétique et la partie numérique
        data[['Alpha', 'Numeric']] = data['ArgName'].str.extract(r'([A-Za-z]+)(\d+)')
        # Convertir la partie numérique en nombre
        data['Numeric'] = data['Numeric'].astype(int)
        # Trier les données par la partie alphabétique et la partie numérique
        data_sorted = data.sort_values(by=['Alpha', 'Numeric'])
        # Réassembler les parties alphabétiques et numériques pour obtenir la colonne 'ArgName' triée
        data_sorted['ArgName'] = data_sorted['Alpha'] + data_sorted['Numeric'].astype(str)

        features_columns = ['ArgBasicWeight', 'FinalWeight']
        features_data = data_sorted[features_columns]
        features_tensor = torch.tensor(features_data.values, dtype=torch.float32)

        # Construction de la matrice d'adjacence pour les arguments (matrice nulle)
        num_nodes = len(data_sorted)
        adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

    else:
        raise FileNotFoundError(f"Le fichier {file_path_arguments} n'existe pas.")

    # Chargement des données d'entailments et mise à jour de la matrice d'adjacence
    if os.path.exists(file_path_entailments):
        entailments_data = pd.read_csv(file_path_entailments)

        # Créer une carte pour mapper les noms des nœuds aux indices
        node_to_index = {node_name: index for index, node_name in enumerate(data_sorted['ArgName'])}

        # Utiliser cette carte pour indexer les nœuds dans les données d'entailments
        for row in entailments_data.itertuples():
            relation = row.Entailment
            node_a = row.Arg1
            node_b = row.Arg2

            if node_a in node_to_index and node_b in node_to_index:
                index_a = node_to_index[node_a]
                index_b = node_to_index[node_b]

                if relation == 'sup':
                    adjacency_matrix[index_a, index_b] = 1.0
                elif relation == 'att':
                    adjacency_matrix[index_a, index_b] = -1.0
            else:
                print(
                    f"Erreur: Noeud non trouvé dans la carte pour la relation '{relation}' entre les noeuds '{node_a}' et '{node_b}' dans le graph '{graph_id}'")

    else:
        print(f"Le fichier {file_path_entailments} n'existe pas.")

    return features_tensor, adjacency_matrix

def build_data_object(graph_id):
    # Charge les données du graphe
    features_matrix, adjacency_matrix = from_file_to_matrix(graph_id)

    # Sélectionner uniquement la première colonne des caractéristiques
    x = features_matrix[:, 0].reshape(-1, 1)  # Reshape pour conserver les dimensions appropriées

    # Convertir la matrice d'adjacence en format coo_matrix
    adjacency_coo = coo_matrix(adjacency_matrix.numpy())

    # Convertir la matrice d'adjacence en edge_index et edge_attribute
    edge_index, edge_attr = utils.from_scipy_sparse_matrix(adjacency_coo)

    # Extraire les y_labels à partir de la deuxième composante de features_matrix
    y_labels = features_matrix[:, 1]  # Index 1 pour la deuxième composante du couple

    # Créer l'objet Data et ajouter l'attribut graph_id
    data = Data(x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y_labels,
                graph_id=graph_id)  # Ajout de l'attribut graph_id

    return data


def load_and_split_data(train_size=0.7, val_size=0.15, seed=42):
    """
    Charge les données et les divise en ensembles d'entraînement, de validation et de test.
    """
    import random
    import torch
    from tqdm import tqdm

    # Définir la graine pour la reproductibilité
    random.seed(seed)

    # Charger les identifiants des graphes
    #all_graph_ids = [f'bag_{i * 100}_{j}' for i in range(1, 3) for j in range(100)]
    all_graph_ids = [f'bag_{3000}_{j}' for j in range(1,10)]

    # Liste des identifiants des graphes divergents à retirer
    graphs_to_exclude = ['bag_400_97', 'bag_600_54', 'bag_1000_17', 'bag_1900_95', 'bag_300_52', 'bag_1100_7', 'bag_1800_36', 'bag_500_30', 'bag_100_59', 'bag_100_3', 'bag_400_87', 'bag_100_97', 'bag_2500_58', 'bag_500_35', 'bag_700_93', 'bag_200_4', 'bag_500_17', 'bag_100_13', 'bag_600_63', 'bag_300_4']

    # Retirer les identifiants des graphes divergents de la liste all_graph_ids
    all_graph_ids = [graph_id for graph_id in all_graph_ids if graph_id not in graphs_to_exclude]

    # Mélanger les identifiants
    random.shuffle(all_graph_ids)

    # Créer la liste de données
    data_list = []

    print("Chargement des données :")
    for graph_id in tqdm(all_graph_ids):
        data = build_data_object(graph_id)
        num_nodes = data.y.size(0)

        # Créer les masques individuels pour chaque nœud
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # Attribuer un masque à chaque nœud en fonction des proportions définies
        for i in range(num_nodes):
            rand = random.random()
            if rand < train_size:
                train_mask[i] = True
            elif rand < train_size + val_size:
                val_mask[i] = True
            else:
                test_mask[i] = True

        # Assigner les masques à l'objet data
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        # Ajouter data à data_list
        data_list.append(data)

    return data_list

