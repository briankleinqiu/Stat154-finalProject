import numpy as np
import scipy 

def load_csv(filename):
    """loads list of csv files into numpy arrays"""
    return(np.genfromtxt(filename, delimiter = ",", dtype = int, usecols = 1, skip_header = 1))

xg = load_csv("xgboost_submission.csv")
xg_tree = load_csv("xgtree_submission.csv")
log = load_csv("logistic_submission.csv")
naive = load_csv("naivebayes_submission.csv") 

print(xg)
print(xg_tree)
print(log)
print(naive)

print(np.corrcoef([xg, xg_tree, log, naive]))
