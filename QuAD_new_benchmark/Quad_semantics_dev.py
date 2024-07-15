import csv
import numpy as np
import pandas as pd
import time
import os
import sys

def argumentt(id,degrees,graph):
    arguments= pd.read_csv('arguments/' + graph+'_arguments.csv')
    entailments=pd.read_csv('entailments/' + graph+'_entailment.csv')
    index=arguments[arguments['ArgName']==id].index.values[0]
    basic_weights=arguments[('ArgBasicWeight')]
    basic_weight = basic_weights[index]
    attackers=[]
    supporters=[]
    deg_attackers=[]
    deg_supporters=[]
    for j in range(len(entailments)):
            if entailments['Arg2'][j]==id and (entailments['Entailment'][j]== 'sup'):
                supporters.append(entailments['Arg1'][j])
                index=arguments[arguments['ArgName']==entailments['Arg1'][j]].index.values[0]
                deg_supporters.append(degrees[1][index])
            elif entailments['Arg2'][j]==id and (entailments['Entailment'][j]== 'att'):
                attackers.append(entailments['Arg1'][j])
                index=arguments[arguments['ArgName']==entailments['Arg1'][j]].index.values[0]
                deg_attackers.append(degrees[1][index])
    liste = [id,basic_weight, attackers, deg_attackers, supporters, deg_supporters]
    return liste

# Quad takes the name of the graph, opens the corresponding arguments and entailments CSV files,
# and calculates the final weight for each argument using the Quad semantics
def Quad(graph, max_iterations=10, threshold=0.001):
    # Créez les répertoires s'ils n'existent pas
    if not os.path.exists('final_weights'):
        os.mkdir('final_weights')

    if not os.path.exists('times'):
        os.mkdir('times')

    
    if not os.path.exists('generations'):
        os.mkdir('generations')

    # Read argument and entailment data from CSV files
    arguments = pd.read_csv('arguments/' + graph + '_arguments.csv')
    entailments = pd.read_csv('entailments/' + graph + '_entailment.csv')

    
    # Create a copy of the arguments DataFrame
    A = arguments.copy()
    
    # Initialize lists and variables
    degrees = [A.loc[:, ('ArgName')], A.loc[:, ('ArgBasicWeight')]]
    is_changed = []
    start_time = time.time()
    iteration_count = 0
    generation_data = []
    
    # Initialize the is_changed list to track whether an argument's weight has changed
    for i in range(len(arguments['ArgName'])):
        is_changed.append(True)
        #for each argument, we calculate its degree based on the degrees calculated in the previous step, we stop once the weights don't change anymore, which means that is_changed contains only False values

    # Iterate through the arguments to calculate their final weights
    while iteration_count < max_iterations and True in is_changed:
        iteration_count += 1  # Increment the iteration count
        
        B = degrees.copy()
        for i in range(len(arguments['ArgName'])):
            
            argument = argumentt(arguments['ArgName'][i], B, graph)
            
            # Calculate the degree (deg) for each argument
            if not(argument[2]) and not(argument[5]):
                deg = argument[1]
            else:
                deg_attackers1 = []
                deg_supporters1 = []
                
                # Calculate the degree for attackers and supporters
                if len(argument[3]) != 0:
                    for h in range(len(argument[3])):
                        deg_attackers1.append(1 - argument[3][h])
                    fa = argument[1] * np.prod(deg_attackers1)
                else:
                    fa = 0

                if len(argument[5]) != 0:
                    for h in range(len(argument[5])):
                        deg_supporters1.append(1 - argument[5][h])
                    fb = 1 - ((1 - argument[1]) * np.prod(deg_supporters1))
                else:
                    fb = 0

                # Determine the degree (deg) based on attackers and supporters
                if fa == 0 and fb != 0:
                    deg = fb
                elif fb == 0 and fa != 0:
                    deg = fa
                else:
                    deg = (fa + fb) / 2

            # Check if the calculated degree (deg) is different from the argument's original basic weight
            if abs(deg - A.loc[i, ('ArgBasicWeight')]) > threshold:
                A.loc[i, ('ArgBasicWeight')] = deg
                is_changed[i] = True
            else:
                is_changed[i] = False
        # Ajouter la génération actuelle au DataFrame
        generation_data.append(A['ArgBasicWeight'].tolist())


    # Create a list to store argument names and their final weights
    data = []
    for h in range(len(degrees[0])):
        data.append([degrees[0][h], degrees[1][h]])
    # Convert it into dataframe
    df = pd.DataFrame(data)
    df.columns = ['ArgName', 'FinalWeight']

    # Create a DataFrame with the argument names,their final weights and the execution time
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate execution time
    # Create a DataFrame to store the execution time
    time_df = pd.DataFrame(data={'ExecutionTime': [execution_time], 'IterationCount': [iteration_count]})
    generation_df = pd.DataFrame(generation_data)

    # Record the df in a CSV.file# Enregistrez le DataFrame contenant les générations de poids dans un fichier CSV
    generation_df.to_csv('generations/' + graph + '_generation_weights.csv', index=False)
    df.to_csv('final_weights/' + graph + '_final_weights.csv', index=False)
    time_df.to_csv('times/' + graph + '_time.csv', index=False)



    return 0 #there was no problem of divergence

if __name__ == "__main__":
    Quad(sys.argv[1])
