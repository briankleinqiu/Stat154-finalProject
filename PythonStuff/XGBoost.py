from __future__ import division
import numpy as np
import xgboost as xgb

X_train = np.load("Data/X_train.npy")[()]
X_holdout = np.load("Data/X_holdout.npy")[()]
y_train = np.load("Data/y_train.npy")[()]
y_holdout = np.load("Data/y_holdout.npy")[()]

d_train = xgb.DMatrix(X_train, y_train)
d_holdout = xgb.DMatrix(X_holdout)
params = {"booster":"gblinear", "objective":"binary:logistic"}

xg_model = xgb.train(params, d_train)
yhat = xg_model.predict(d_holdout)
print(yhat)
print(len(yhat[yhat > 1]))
yhat[yhat >= .5] = 1
yhat[yhat < .5] = 0  
accuracy = (sum(yhat == y_holdout))/len(yhat)
print(accuracy)

