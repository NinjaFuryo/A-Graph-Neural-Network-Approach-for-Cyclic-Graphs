import numpy as np
from data_loader import from_file_to_matrix
import os
import sys
import pandas as pd
import torch
from tqdm import tqdm

def find_non_attacked_nodes(adjacency):
    """
    Trouve les noeuds qui ne sont pas attaqués dans un graphe.

    Args:
    adjacency (np.array): Matrice d'adjacence du graphe.

    Returns:
    list: Liste des indices des noeuds qui ne sont pas attaqués.
    """
    non_attacked_nodes = []
    num_nodes = adjacency.shape[0]

    for i in range(num_nodes):
        is_attacked = False
        for j in range(num_nodes):
            if adjacency[j, i] == -1:
                is_attacked = True
                break
        if not is_attacked:
            non_attacked_nodes.append(i)

    return non_attacked_nodes
def find_non_supported_nodes(adjacency):
    """
    Cette fonction identifie les noeuds qui ne sont pas supportés.

    Args:
    adjacency (np.array): Matrice d'adjacence du graphe où 1 indique un support.

    Returns:
    list: Liste des indices des noeuds qui ne sont pas supportés.
    """
    num_nodes = adjacency.shape[0]
    non_supported_nodes = []

    for i in range(num_nodes):
        is_supported = False
        for j in range(num_nodes):
            if adjacency[j, i] == 1:
                is_supported = True
                break
        if not is_supported:
            non_supported_nodes.append(i)

    return non_supported_nodes

def find_alpha_nodes(adjacency):
    """
    Cette fonction identifie les noeuds qui ne sont ni attaqués ni supportés.

    Args:
    adjacency (np.array): Matrice d'adjacence du graphe.

    Returns:
    list: Liste des indices des noeuds neutres (ni attaqués, ni supportés).
    """
    non_attacked_nodes = find_non_attacked_nodes(adjacency)
    non_supported_nodes = find_non_supported_nodes(adjacency)
    # Intersection des deux listes pour trouver les noeuds qui ne sont ni attaqués ni supportés
    alpha_nodes = list(set(non_attacked_nodes) & set(non_supported_nodes))
    return alpha_nodes

def find_attackers(node_index, adjacency):
    """
    Identifie les attaquants d'un noeud spécifique.

    Args:
    node_index (int): L'indice du noeud cible.
    adjacency (np.array): La matrice d'adjacence du graphe où -1 indique une attaque.

    Returns:
    list: Liste des indices des noeuds qui attaquent le noeud cible.
    """
    num_nodes = adjacency.shape[0]
    attackers = []
    # Parcourir toutes les lignes de la colonne correspondant à node_index
    for i in range(num_nodes):
        if adjacency[i, node_index] == -1:
            attackers.append(i)
    return attackers

def find_supporters(node_index, adjacency):
    """
    Identifie les supporters d'un noeud spécifique.

    Args:
    node_index (int): L'indice du noeud cible.
    adjacency (np.array): La matrice d'adjacence du graphe où 1 indique un support.

    Returns:
    list: Liste des indices des noeuds qui supportent le noeud cible.
    """
    num_nodes = adjacency.shape[0]
    supporters = []
    # Parcourir toutes les lignes de la colonne correspondant à node_index
    for i in range(num_nodes):
        if adjacency[i, node_index] == 1:
            supporters.append(i)
    return supporters

def group_similar_nodes(features):
    """
    Groups nodes with similar first scores in their feature matrix,
    according to a delta difference threshold. Only groups containing
    at least two nodes.

    Args:
    features (np.array): Node feature matrix.
    delta (float): Threshold of acceptable difference between the first scores to consider two nodes as similar.

    Returns:
    list: List of index lists of nodes with similar first scores, with each list containing at least two nodes.
    """
    num_nodes = features.shape[0]
    groups = {}
    visited = np.zeros(num_nodes, dtype=bool)

    for i in range(num_nodes):
        if not visited[i]:
            current_group = [i]
            visited[i] = True
            for j in range(i + 1, num_nodes):
                if not visited[j]:
                    # Comparaison basée uniquement sur la première colonne (premier score)
                    if features[i, 0] == features[j, 0]:
                        current_group.append(j)
                        visited[j] = True
            if len(current_group) > 1:  # Ajouter le groupe seulement s'il contient au moins deux noeuds
                group_id = len(groups) + 1
                groups[group_id] = current_group

    return list(groups.values())

def score_list(node_indices,features):
    return [features[node, 0] for node in node_indices]

def scores_within_delta(scores_a, scores_b, delta=0.02):
    for score_a, score_b in zip(scores_a, scores_b):
        if abs(score_a - score_b) > delta:
            return False
    return True

def check_features_equality(node_indices, features, delta=0.02):
    """
    Vérifie si tous les noeuds dans une liste ont des features similaires à un delta près.
    Args:
    node_indices (list): Liste des indices des noeuds.
    features (np.array): Matrice de features des noeuds.
    delta (float): Tolérance pour la comparaison des features.

    Returns:
    bool: True si tous les noeuds ont des features similaires à un delta près, sinon False.
    """
    if not node_indices:
        return True
    reference_features = features[node_indices[0]]

    for node_index in node_indices[1:]:
        node_features = features[node_index]
        if not scores_within_delta(reference_features, node_features, delta):
            return False
    return True


def lists_almost_equal(list_a, list_b):
    """
    Vérifie si deux listes sont identiques à un élément près, quel que soit l'ordre.

    Args:
    list_a (list): Première liste d'indices.
    list_b (list): Seconde liste d'indices.

    Returns:
    tuple: (bool, tuple) où le booléen indique si les listes sont équivalentes à un élément près,
           et le tuple contient la paire d'éléments uniques si True, sinon vide.
    """
    # Convertir les listes en ensembles pour faciliter les opérations
    set_a = set(list_a)
    set_b = set(list_b)

    if len(list_a) == len(list_b):
        # Vérifier si les deux ensembles sont identiques
        if set_a == set_b:
            return (True, tuple())

        # Calculer la différence symétrique pour trouver les éléments uniques dans chaque ensemble
        sym_diff = sorted(set_a.symmetric_difference(set_b))  # Tri des éléments de la différence symétrique

        # Si la différence symétrique contient exactement deux éléments, alors ils sont équivalents à un élément près
        if len(sym_diff) == 2:
            # Vérifier l'ordre des éléments dans la différence symétrique
            if sym_diff[0] in list_a:
                first_element = sym_diff[0]
                second_element = sym_diff[1]
            else:
                first_element = sym_diff[1]
                second_element = sym_diff[0]

            return (True, (first_element, second_element))

    # Dans tous les autres cas, retourner False
    return (False, tuple())


def check_injection(list_A, list_B, features, delta=0.02):
    """
    Vérifie si une injection de la liste A dans la liste B satisfait une condition sur les scores de première feature.

    Args:
    list_A (list): Liste d'indices de la première liste.
    list_B (list): Liste d'indices de la deuxième liste.
    features (np.array): Matrice de features des noeuds.

    Returns:
    bool: True si la condition est satisfaite, sinon False.
    """
    # Vérifie si la longueur de list_A est inférieure ou égale à celle de list_B
    if len(list_A) >= len(list_B):
        return False

    # Trie les indices de list_A par score de première feature (ordre décroissant)
    sorted_list_A = sorted(list_A, key=lambda x: features[x, 1], reverse=True)

    # Trie les indices de list_B par score de première feature (ordre décroissant)
    sorted_list_B = sorted(list_B, key=lambda x: features[x, 1], reverse=True)

    # Vérifie la condition d'injection
    for i, a in enumerate(sorted_list_A):
        # Vérifie si le score de deuxieme feature de a est inférieur ou égal à celui de son image dans list_B
        if features[a, 1] >= features[sorted_list_B[i], 1] - delta :
            return False

    return True



################################################################

# Principe

#pincipe N3
def check_directionality(features, adjacency , delta=0.02):
    """
    Vérifie pour chaque paire de noeuds ayant des premiers scores similaires si:
    1. Ils ont les mêmes attaquants et les mêmes supporters.
    2. La différence de leur deuxième score est inférieure ou égale à delta.

    Args:
    features (np.array): Matrice de features des noeuds.
    adjacency (np.array): Matrice d'adjacence du graphe.
    delta (float): Seuil de différence acceptable pour la deuxième feature.

    Returns:
    bool: True si toutes les paires valides répondent à ces critères, sinon False.
    """
    # Trouver les groupes de noeuds avec des scores similaires à la première feature
    groups = group_similar_nodes(features)

    # Examiner chaque groupe pour des paires de noeuds
    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                node1, node2 = group[i], group[j]

                # Utiliser les fonctions existantes pour trouver les attaquants et les supporters
                attackers1 = set(find_attackers(node1, adjacency))
                attackers2 = set(find_attackers(node2, adjacency))
                supporters1 = set(find_supporters(node1, adjacency))
                supporters2 = set(find_supporters(node2, adjacency))

                # Vérifier si les attaquants et les supporters sont identiques
                if attackers1 == attackers2 and supporters1 == supporters2:
                    # Vérifier la différence de la deuxième feature
                    if abs(features[node1, 1] - features[node2, 1]) > delta:
                        return False

    return True

#principe N°4
def check_equivalence(features, adjacency, delta=0.02):
    """
    Fonction complète vérifiant plusieurs critères de similitude et d'équivalence entre les noeuds.
    """
    groups = group_similar_nodes(features)

    # Examiner chaque groupe pour des paires de noeuds
    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                node_a, node_b = group[i], group[j]
                attackers_a = find_attackers(node_a, adjacency)
                attackers_b = find_attackers(node_b, adjacency)
                supporters_a = find_supporters(node_a, adjacency)
                supporters_b = find_supporters(node_b, adjacency)

                if len(attackers_a) == len(attackers_b) and len(supporters_a) == len(supporters_b):
                    # Vérification de l'égalité des scores de deuxième feature des attaquants et des supporters
                    if scores_within_delta(sorted(score_list(attackers_a, features)),
                                           sorted(score_list(attackers_b, features)), delta) and \
                       scores_within_delta(sorted(score_list(supporters_a, features)),
                                           sorted(score_list(supporters_b, features)), delta):
                        # Vérification de la différence entre les scores de deuxième feature des noeuds
                        if abs(features[node_a, 1] - features[node_b, 1]) > delta:
                            return False
    return True


#principe N°5
def check_stability(features, adjacency, delta=0.02):
    """
    Vérifie si les deux scores de chaque noeud neutre sont presque égaux à un seuil spécifié et
    si tous les noeuds neutres respectent cette condition.

    Args:
    features (np.array): Matrice de features des noeuds.
    adjacency (np.array): Matrice d'adjacence du graphe.
    delta (float): Seuil de différence acceptable entre les deux scores.

    Returns:
    bool: True si tous les noeuds neutres ont des scores qui diffèrent par moins de ou égal à delta, sinon False.
    """
    alpha_nodes = find_alpha_nodes(adjacency)
    for node in alpha_nodes:
        score_diff = abs(features[node, 0] - features[node, 1])
        if score_diff > delta:
            return False  # Dès qu'un noeud ne respecte pas la condition, retourne False
    return True

#principe N°6
def check_neutrality(features, adjacency, delta=0.02):
    """
    Vérifie pour chaque noeud si tous ses attaquants et supporters ont un score de deuxième feature de zéro,
    et si la différence entre les scores de première et de deuxième feature du noeud est supérieure à delta.

    Args:
    features (np.array): Matrice de features des noeuds.
    adjacency (np.array): Matrice d'adjacence du graphe.
    delta (float): Seuil de différence acceptable entre les deux features.

    Returns:
    bool: True si aucun noeud ne viole la neutralité, sinon False.
    """
    num_nodes = features.shape[0]

    for node in range(num_nodes):
        # Trouver les attaquants et les supporters du noeud
        attackers = find_attackers(node, adjacency)
        supporters = find_supporters(node, adjacency)

        # Vérifier si tous les attaquants et supporters ont un score de deuxième feature de zéro
        related_nodes = attackers + supporters
        if all(features[n, 1] == 0 for n in related_nodes):
            # Vérifier si la différence entre les scores de première et de deuxième feature est supérieure à delta
            if abs(features[node, 0] - features[node, 1]) > delta:
                return False

    return True

#principe N°7
def check_monotony(features, adjacency, delta=0.02):
    """
    Vérifie pour chaque paire de noeuds ayant des premiers scores similaires si :
    1. Les attaquants de a sont égaux ou inclus à ceux de b.
    2. Les supporters de b sont égaux ou inclus à ceux de a.
    3. La seconde feature de a est inférieure à celle de b modulo à plus ou moins delta.

    Args:
    features (np.array): Matrice de features des noeuds.
    adjacency (np.array): Matrice d'adjacence du graphe.
    delta (float): Tolérance pour la comparaison des secondes features.

    Returns:
    bool: True si tous les noeuds vérifient ces conditions pour toutes leurs paires valides, sinon False.
    """
    groups = group_similar_nodes(features)

    def is_subset(set_a, set_b):
        return set_a.issubset(set_b)

    # Examiner chaque groupe pour des paires de noeuds
    for group in groups:
        for i in range(len(group)):
            for j in range(len(group)):
                if i != j:  # Assurer que nous ne comparons pas le noeud avec lui-même
                    node_a, node_b = group[i], group[j]

                    # Utiliser les fonctions existantes pour trouver les attaquants et les supporters
                    attackers_a = set(find_attackers(node_a, adjacency))
                    attackers_b = set(find_attackers(node_b, adjacency))
                    supporters_a = set(find_supporters(node_a, adjacency))
                    supporters_b = set(find_supporters(node_b, adjacency))

                    # Vérifier les conditions d'inclusion
                    if is_subset(attackers_a, attackers_b) and is_subset(supporters_b, supporters_a):
                        # Vérifier la condition sur la seconde feature
                        if (features[node_a, 1] + delta < features[node_b, 1]):
                            #print(node_a)
                            #print(node_b)
                            return False

    return True

graph_id='bag_3000_0'
features, adjacency = from_file_to_matrix(graph_id)
prediction_file = f"predictions/{graph_id}_prediction.csv"
predictions_df = pd.read_csv(prediction_file, header=0)
predictions = predictions_df['Prediction'].values

# Mettre à jour la deuxième colonne de la matrice features avec les prédictions
features[:, 1] = torch.tensor(predictions)

if check_monotony(features,adjacency):
    print('yes')
else:
    print('nah')
#principe N°8
def check_reinforcement(features, adjacency, delta=0.02):
    """
    Effectue des vérifications complexes sur les relations de graphes basées sur des conditions spécifiques.

    Args:
    graph_id (int): Identifiant du graph pour charger les matrices features et adjacency.
    delta (float): Tolérance pour la comparaison des features.

    Returns:
    bool: True si toutes les conditions sont satisfaites, sinon False.
    """
    groups = group_similar_nodes(features)  # Trouve les groupes de noeuds avec des premiers scores similaires

    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                node_a, node_b = group[i], group[j]

                # Obtenir les listes d'attaquants et de supporters pour chaque noeud
                attackers_a = find_attackers(node_a, adjacency)
                attackers_b = find_attackers(node_b, adjacency)
                supporters_a = find_supporters(node_a, adjacency)
                supporters_b = find_supporters(node_b, adjacency)

                # Vérifier les attaquants et supporters avec lists_almost_equal
                result_attackers, pair_attackers = lists_almost_equal(attackers_a, attackers_b)

                result_supporters, pair_supporters = lists_almost_equal(supporters_a, supporters_b)


                if result_attackers and result_supporters:
                    # Extrait les éléments uniques
                    if pair_attackers and pair_supporters:
                        c, d = pair_attackers
                        e, f = pair_supporters
                        # Vérifier les conditions sur les scores de la première feature
                        condition1 = (features[c, 1] > features[d, 1] + delta and features[e, 1] < features[f, 1] - delta)
                        condition2 = (features[c, 1] < features[d, 1] - delta and features[e, 1] > features[f, 1] + delta)

                        # Vérifier les conditions sur les scores de la deuxième feature
                        if condition1 and (features[node_b, 1] + delta < features[node_a, 1]):
                            return False
                        if condition2 and (features[node_a, 1] + delta < features[node_b, 1]):
                            return False

    return True

#principe N°9
#def check_resilience()

#principe N°10
def check_franklin(features, adjacency, delta=0.02):
    """
    Vérifie les relations entre les features des noeuds du graphe.

    Args:
    graph_id (int): Identifiant du graphe pour charger les matrices features et adjacency.
    delta (float): Tolérance pour la comparaison des features.

    Returns:
    bool: True si toutes les conditions sont satisfaites pour toutes les paires de noeuds, sinon False.
    """

    groups = group_similar_nodes(features)  # Trouve les groupes de noeuds avec des premiers scores similaires

    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                node_a, node_b = group[i], group[j]

                # Obtient les listes d'attaquants et de supporters pour chaque noeud
                attackers_a = find_attackers(node_a, adjacency)
                attackers_b = find_attackers(node_b, adjacency)
                supporters_a = find_supporters(node_a, adjacency)
                supporters_b = find_supporters(node_b, adjacency)

                # Vérifie si les attaquants et supporters sont presque identiques avec un delta
                if (lists_almost_equal(attackers_a, attackers_b) and
                    lists_almost_equal(supporters_a, supporters_b)):

                    # Trouve l'élément unique dans les attaquants et les supporters
                    unique_attacker = list(set(attackers_a).symmetric_difference(set(attackers_b)))
                    unique_supporter = list(set(supporters_a).symmetric_difference(set(supporters_b)))

                    if len(unique_attacker) == 1 and len(unique_supporter) == 1:
                        x, y = unique_attacker[0], unique_supporter[0]

                        # Vérifie si le score de première feature de x est égal au score de y
                        if features[x, 1] == features[y, 1]:
                            # Vérifie si le score de deuxième feature de a est supérieur à celui de b avec un delta
                            if features[node_a, 1] > features[node_b, 1] + delta:
                                return False

    return True

def check_weakening(features, adjacency, delta=0.02):
    """
    Vérifie les conditions spécifiques pour chaque noeud du graphe.

    Args:
    graph_id (int): Identifiant du graphe pour charger les matrices features et adjacency.
    delta (float): Tolérance pour la comparaison des features.

    Returns:
    bool: True si toutes les conditions sont satisfaites pour tous les noeuds, sinon False.
    """

    num_nodes = features.shape[0]  # Nombre total de noeuds dans le graphe

    for node in range(num_nodes):
        supporters = find_supporters(node, adjacency)
        attackers = find_attackers(node, adjacency)

        # Vérifie la condition d'injection pour les supporters et les attaquants du noeud
        if check_injection(supporters, attackers, features, delta):
            # Vérifie si le score de première feature de a est plus grand que score de deuxième feature de a + delta
            if features[node, 0] < features[node, 1] - delta:
                return False

    return True

def check_strengthening(features, adjacency, delta=0.02):
    """
    Vérifie les conditions spécifiques pour chaque noeud du graphe.

    Args:
    graph_id (int): Identifiant du graphe pour charger les matrices features et adjacency.
    delta (float): Tolérance pour la comparaison des features.

    Returns:
    bool: True si toutes les conditions sont satisfaites pour tous les noeuds, sinon False.
    """

    num_nodes = features.shape[0]  # Nombre total de noeuds dans le graphe

    for node in range(num_nodes):
        supporters = find_supporters(node, adjacency)
        attackers = find_attackers(node, adjacency)

        # Vérifie la condition d'injection pour les supporters et les attaquants du noeud
        if check_injection(attackers, supporters, features):
            # Vérifie si le score de première feature de a est plus grand que score de deuxième feature de a + delta
            if features[node, 1] < features[node, 0] - delta:
                return False

    return True


import torch
def run_tests_on_all_graphs(all_graph_ids, delta=0.02):
    directionality_counter = 0
    equivalence_counter = 0
    stability_counter = 0
    neutrality_counter = 0
    monotony_counter = 0
    reinforcement_counter = 0
    franklin_counter = 0
    weakening_counter = 0
    strengthening_counter = 0

    # Utilisation de tqdm pour afficher une barre de progression
    with tqdm(total=len(all_graph_ids)) as pbar:
        for graph_id in all_graph_ids:
            # Charger les données features et adjacency
            features, adjacency = from_file_to_matrix(graph_id)

            # Charger les prédictions depuis le fichier CSV
                prediction_file = f"predictions/{graph_id}_prediction.csv"
                predictions_df = pd.read_csv(prediction_file, header=0)
                predictions = predictions_df['Prediction'].values

                # Mettre à jour la deuxième colonne de la matrice features avec les prédictions
                features[:, 1] = torch.tensor(predictions)

            if check_directionality(features, adjacency, delta=delta):
                directionality_counter += 1

            if check_equivalence(features, adjacency, delta=delta):
                equivalence_counter += 1

            if check_stability(features, adjacency, delta=delta):
                stability_counter += 1

            if check_neutrality(features, adjacency, delta=delta):
                neutrality_counter += 1

            if check_monotony(features, adjacency, delta=delta):
                monotony_counter += 1

            if check_reinforcement(features, adjacency, delta=delta):
                reinforcement_counter += 1

            if check_franklin(features, adjacency, delta=delta):
                franklin_counter += 1

            if check_weakening(features, adjacency, delta=delta):
                weakening_counter += 1

            if check_strengthening(features, adjacency, delta=delta):
                strengthening_counter += 1

            # Mise à jour de la barre de progression
            pbar.update(1)

    # Affichage des résultats
    print("Directionality Counter:", directionality_counter)
    print("Equivalence Counter:", equivalence_counter)
    print("Stability Counter:", stability_counter)
    print("Neutrality Counter:", neutrality_counter)
    print("Monotony Counter:", monotony_counter)
    print("Reinforcement Counter:", reinforcement_counter)
    print("Franklin Counter:", franklin_counter)
    print("Weakening Counter:", weakening_counter)
    print("Strengthening Counter:", strengthening_counter)

    # Renvoi des compteurs
    return (directionality_counter, equivalence_counter, stability_counter, neutrality_counter,
            monotony_counter, reinforcement_counter, franklin_counter, weakening_counter, strengthening_counter)


def main(mode=0):
    """
    Fonction principale pour exécuter les tests sur les graphes.

    Args:
    mode (int): Le mode de subdivision des graphes sur les 8 machines disponibles. Valeurs de 0 à 7.
                Chaque valeur divise les graphes en sous-ensembles en fonction du reste de la division par 30 de leur indice.
                Par exemple, si mode=0, les graphes d'indice 0, 30, 60, etc., seront exécutés sur cette machine.
                Si mode=1, les graphes d'indice 1, 31, 61, etc., seront exécutés sur cette machine, et ainsi de suite.

    """
    # Définition de tous les graphes à tester
    # Générer tous les identifiants de graphes classiquement
    #all_graph_ids = [f'bag_{i * 100}_{j}' for i in range(1, 31) for j in range(100)]
    all_graph_ids = [f'bag_{100}_{j}' for j in range(50)]

    # Liste des identifiants des graphes divergents à retirer
    graphs_to_exclude = ['bag_400_97', 'bag_600_54', 'bag_1000_17', 'bag_1900_95', 'bag_300_52', 'bag_1100_7',
                         'bag_1800_36', 'bag_500_30', 'bag_100_59', 'bag_100_3', 'bag_400_87', 'bag_100_97',
                         'bag_2500_58', 'bag_500_35', 'bag_700_93', 'bag_200_4', 'bag_500_17', 'bag_100_13',
                         'bag_600_63', 'bag_300_4']

    # Retirer les identifiants des graphes divergents de la liste all_graph_ids
    all_graph_ids = [graph_id for graph_id in all_graph_ids if graph_id not in graphs_to_exclude]


    # Exécution des tests sur tous les graphes avec un delta de 0.02
    counters = run_tests_on_all_graphs(all_graph_ids, delta=0.02)

    # Affichage des résultats
    print("Directionality Counter:", counters[0])
    print("Equivalence Counter:", counters[1])
    print("Stability Counter:", counters[2])
    print("Neutrality Counter:", counters[3])
    print("Monotony Counter:", counters[4])
    print("Reinforcement Counter:", counters[5])
    print("Franklin Counter:", counters[6])
    print("Weakening Counter:", counters[7])
    print("Strengthening Counter:", counters[8])

    # Enregistrement des résultats dans un fichier
    result_file_name = f"prop_stat/result_mode_{mode}.csv"
    result_data = {
        "Directionality Counter": [counters[0]],
        "Equivalence Counter": [counters[1]],
        "Stability Counter": [counters[2]],
        "Neutrality Counter": [counters[3]],
        "Monotony Counter": [counters[4]],
        "Reinforcement Counter": [counters[5]],
        "Franklin Counter": [counters[6]],
        "Weakening Counter": [counters[7]],
        "Strengthening Counter": [counters[8]]
    }
    df = pd.DataFrame(result_data)
    df.to_csv(result_file_name, index=False)


import torch


def proba_stat_per_graph(graph_id):
    features, _ = from_file_to_matrix(graph_id)

    feature_0 = features[:, 0]  # Première feature
    feature_1 = features[:, 1]  # Deuxième feature

    # Assurez-vous que les tensors sont de type float
    feature_0 = feature_0.float()
    feature_1 = feature_1.float()

    # Calcul de la différence moyenne entre les deux features
    difference_mean = torch.mean(feature_1 - feature_0)

    # Calcul de la variance de la différence entre les deux features
    difference_variance = torch.var(feature_1 - feature_0)

    return difference_mean.item(), difference_variance.item()



# Appel de la fonction main pour exécuter les tests
#if __name__ == "__main__":
 #   if len(sys.argv) > 1:
  #      mode = int(sys.argv[1])
   #     main(mode)
    #else:
     #   main()












