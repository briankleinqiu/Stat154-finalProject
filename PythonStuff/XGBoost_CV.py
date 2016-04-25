from __future__ import division
import numpy as np
import xgboost as xgb

X_train = np.load("Data/X_train.npy")[()]
X_holdout = np.load("Data/X_holdout.npy")[()]
y_train = np.load("Data/y_train.npy")[()]
y_holdout = np.load("Data/y_holdout.npy")[()]

d_train = xgb.DMatrix(X_train, y_train)
d_holdout = xgb.DMatrix(X_holdout)

params_linear = {
        "booster":"gblinear", 
        "objective":"binary:logistic"
}

xg_linear = xgb.cv(params_linear, d_train, num_boost_round = 10, nfold = 5)
print("LINEAR MODEL")
print(xg_linear)
#.793495 accuracy with nround = 5

params_tree = {
        "booster":"gbtree",
        "objective":"binary:logistic"
        }
xg_tree = xgb.cv(params_tree, d_train, num_boost_round = 10, nfold = 5)
print("TREE MODEL")
print(xg_tree)
#.684774% accuracy with nround = 10
