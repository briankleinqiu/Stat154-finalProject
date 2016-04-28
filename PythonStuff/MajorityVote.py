from __future__ import division
import numpy as np
import scipy 

def load_csv(filename):
    """loads list of csv files into numpy arrays"""
    return(np.genfromtxt(filename, delimiter = ",", dtype = int, usecols = 1, skip_header = 1))

y_holdout = np.load("Data/y_holdout.npy")[()]

xg = load_csv("xgboost_submission.csv")
xg_tree = load_csv("xgtree_submission.csv")
log = load_csv("logistic_submission.csv")
naive = load_csv("naivebayes_submission.csv") 
linearsvm = load_csv("linearsvm_submission.csv")
kernelsvm = load_csv("kernelsvm_submission.csv")
#randomforest = load_csv("randomforest_submission.csv")

holdout_xg = load_csv("holdout_xgboost_submission.csv")
holdout_xg_tree = load_csv("holdout_xgtree_submission.csv")
holdout_log = load_csv("holdout_logistic_submission.csv")
holdout_naive = load_csv("holdout_naivebayes_submission.csv") 
holdout_linearsvm = load_csv("holdout_linearsvm_submission.csv")
holdout_kernelsvm = load_csv("holdout_kernelsvm_submission.csv")
#holdout_randomforest = load_csv("holdout_randomforest_submission.csv")

print(xg)
print(holdout_xg_tree)
print(holdout_log)
print(holdout_naive)
print(holdout_linearsvm)

print(np.corrcoef([
    xg,
    xg_tree, 
    log, 
    naive, 
    linearsvm, 
    kernelsvm
    #randomforest
    ]))
#Therefore xg, logistic, linearsvm all highly correlated (> .95)

models = np.asarray((
        holdout_xg_tree,
        holdout_log,
        holdout_naive,
        holdout_linearsvm,
        holdout_xg,
        holdout_kernelsvm
        ))
weights = np.asarray((1,3,1,1,1,1))
holdout_combined = weights.dot(models)
print(holdout_combined.shape)
holdout_combined = holdout_combined/8
holdout_combined[holdout_combined >= .5] = 1
holdout_combined[holdout_combined < .5] = 0

print("holdout accuracy of majority vote:")
print(sum(holdout_combined == y_holdout)/len(holdout_combined)) #.79024, lower than sparse itself


"""
combined = xg_tree + log + naive
combined = combined/3
combined[combined >= .5] = 1
combined[combined < .5] = 0

id = np.arange(50000) + 1
result =  np.column_stack((id, combined.astype(int)))
with open("majority_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")
"""
