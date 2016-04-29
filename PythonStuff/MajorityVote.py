from __future__ import division
import numpy as np
import scipy 
from sklearn.cross_validation import KFold


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
randomforest = load_csv("randomforest_submission.csv")
perceptron = load_csv("perceptron_submission.csv")



holdout_xg = load_csv("holdout_xgboost_submission.csv")
holdout_xg_tree = load_csv("holdout_xgtree_submission.csv")
holdout_log = load_csv("holdout_logistic_submission.csv")
holdout_naive = load_csv("holdout_naivebayes_submission.csv") 
holdout_linearsvm = load_csv("holdout_linearsvm_submission.csv")
holdout_kernelsvm = load_csv("holdout_kernelsvm_submission.csv")
holdout_randomforest = load_csv("holdout_randomforest_submission.csv")
holdout_perceptron = load_csv("holdout_perceptron_submission.csv")

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
    kernelsvm,
    randomforest,
    perceptron
    ]))
#Therefore xg, logistic, linearsvm all highly correlated (> .95)

models = np.asarray((
    holdout_xg_tree,
    holdout_log,
    holdout_naive,
    holdout_linearsvm,
    holdout_xg,
    holdout_kernelsvm,
    holdout_randomforest,
    holdout_perceptron
    ))

weights = np.asarray((1,4,1,1,1,1,1,1))
holdout_combined = weights.dot(models)
holdout_combined = holdout_combined/sum(weights)
holdout_combined[holdout_combined >= .5] = 1
holdout_combined[holdout_combined < .5] = 0

print("holdout accuracy of majority vote:")
print(sum(holdout_combined == y_holdout)/len(holdout_combined)) #.79636
"""
#CROSS VALIDATION
models = np.asarray((
    holdout_xg,
    holdout_xg_tree,
    holdout_log,
    holdout_naive,
    holdout_linearsvm,
    holdout_kernelsvm,
    holdout_randomforest,
    holdout_perceptron
    ))
weights = np.asarray((1,1,1,1,1,1,1,1))

for i in np.arange(9):
    weights[2] = i
    holdout_combined = weights.dot(models)
    holdout_combined = holdout_combined/sum(weights)
    holdout_combined[holdout_combined >= .5] = 1
    holdout_combined[holdout_combined < .5] = 0
    print(i)
    print(sum(holdout_combined == y_holdout)/len(holdout_combined)) 

#.79636 on cv score for log weight = 4

"""


"""
#Create submission
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
