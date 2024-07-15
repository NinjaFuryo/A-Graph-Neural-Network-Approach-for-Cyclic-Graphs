import os
import pandas as pd

# Spécifiez le chemin vers le répertoire contenant les fichiers arguments
arguments_dir = './arguments'

# Spécifiez le chemin vers le répertoire contenant les fichiers final_weights
final_weights_dir = './final_weights'

# Spécifiez le chemin vers le répertoire où vous souhaitez sauvegarder les fichiers mis à jour
output_dir = './final_arguments_with_weights'

# Créez le répertoire de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Boucle sur XX
for XX in range(1, 31):
    # Boucle sur YY
    for YY in range(0, 100):
        # Construisez le nom du graph
        graph_name = f'bag_{XX * 100}_{YY}'

        # Construisez le chemin complet vers le fichier d'arguments
        argument_path = os.path.join(arguments_dir, f'{graph_name}_arguments.csv')

        # Construisez le chemin complet vers le fichier des final_weights
        final_weights_path = os.path.join(final_weights_dir, f'{graph_name}_final_weights.csv')

        # Vérifiez si les deux fichiers existent avant de les charger
        if os.path.exists(argument_path) and os.path.exists(final_weights_path):
            # Chargez les fichiers d'arguments et final_weights dans des DataFrames
            argument_df = pd.read_csv(argument_path)
            final_weights_df = pd.read_csv(final_weights_path)

            # Fusionnez les DataFrames sur la colonne 'ArgName'
            merged_df = pd.merge(argument_df, final_weights_df[['ArgName', 'FinalWeight']], how='left', on='ArgName')

            # Sauvegardez le DataFrame fusionné dans le répertoire de sortie
            output_path = os.path.join(output_dir, f'{graph_name}_arguments_with_weights.csv')
            merged_df.to_csv(output_path, index=False)
