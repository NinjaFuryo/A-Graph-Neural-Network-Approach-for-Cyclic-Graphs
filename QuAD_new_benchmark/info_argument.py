import csv
import numpy as np
import pandas as pd

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
    