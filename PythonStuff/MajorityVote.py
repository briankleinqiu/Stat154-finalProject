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

holdout_xg_tree = load_csv("holdout_xgtree_submission.csv")
holdout_log = load_csv("holdout_logistic_submission.csv")
holdout_naive = load_csv("holdout_naivebayes_submission.csv") 
holdout_linearsvm = load_csv("holdout_linearsvm_submission.csv")
holdout_kernelsvm = load_csv("holdout_kernelsvm_submission.csv")


print(xg)
print(holdout_xg_tree)
print(holdout_log)
print(holdout_naive)
print(holdout_linearsvm)

print(np.corrcoef([xg, xg_tree, log, naive, linearsvm, kernelsvm]))
#xg and log highly correlated as expected (.95 corr coef) so just use log since slightly better prediction
#linearsvm and log extremely correlated as well (.98 corr coef)
#Therefore xg, logistic, linearsvm all highly correlated

holdout_combined = holdout_xg_tree + holdout_log + holdout_naive + holdout_linearsvm
holdout_combined = holdout_combined/4
holdout_combined[holdout_combined >= .5] = 1
holdout_combined[holdout_combined < .5] = 0

print("holdout accuracy of majority vote:")
print(sum(holdout_combined == y_holdout)/len(holdout_combined)) #.79024, lower than sparse itself


combined = xg_tree + log + naive
combined = combined/3
combined[combined >= .5] = 1
combined[combined < .5] = 0

id = np.arange(50000) + 1
result =  np.column_stack((id, combined.astype(int)))
with open("majority_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")
