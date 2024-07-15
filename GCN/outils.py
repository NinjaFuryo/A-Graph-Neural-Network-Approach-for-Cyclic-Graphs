from scipy.stats import kendalltau as kt
from scipy.stats import rankdata as rd
def kendall_tau(gold, pred):
    rank_gold = rd(gold)
    rank_pred = rd(pred)
    return kt(rank_gold, rank_pred)

