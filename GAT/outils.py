from scipy.stats import kendalltau as kt
from scipy.stats import rankdata as rd
import numpy as np
def kendall_tau(gold, pred):
    rank_gold = rd(gold)
    rank_pred = rd(pred)
    return kt(rank_gold, rank_pred)

def mean_square_error(gold, pred):
    """
    Calcule l'erreur quadratique moyenne (mean square error) entre deux vecteurs.

    Args:
    gold (list or numpy array): Vecteur de valeurs de référence.
    pred (list or numpy array): Vecteur de valeurs prédites.

    Returns:
    float: Erreur quadratique moyenne (mean square error).
    """
    return np.mean((np.array(gold) - np.array(pred)) ** 2)

vec_one=[3,1,6,5,2,4]
vec_two=[4,2,5,6,1,3]
print(kendall_tau(vec_one,vec_two))

def calc(a,b,it=40,delta=0.01):
    wa=float(a)
    wb=float(b)
    newa=float(a)
    newb=float(b)
    counter = 0
    while counter < it:
        if (abs(newa-wa) <delta and abs(newb-wb)<delta and counter>0):
            print(newa,newb)
            return (newa,newb)
        temp_a=(wa*(1-newb))
        temp_b=(wb*(1-newa))
        print(temp_a)
        wa=newa
        wb=newb
        newa=temp_a
        newb=temp_b
        counter += 1
    print('overtime')
    print(newa,newb)
    return (newa,newb)

