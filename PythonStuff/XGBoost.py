from __future__ import division
import numpy as np
import xgboost as xgb

X_train = np.load("Data/X_train.npy")[()]
X_holdout = np.load("Data/X_holdout.npy")[()]
y_train = np.load("Data/y_train.npy")[()]
y_holdout = np.load("Data/y_holdout.npy")[()]
X_test = np.load("Data/X_test.npy")[()]



######LINEAR#######
d_train = xgb.DMatrix(X_train, y_train)
d_holdout = xgb.DMatrix(X_holdout)
d_test = xgb.DMatrix(X_test)
params = {"booster":"gblinear", "objective":"binary:logistic"}

xg_model = xgb.train(params, d_train, num_boost_round = 5)
yhat = xg_model.predict(d_holdout)
yhat[yhat >= .5] = 1
yhat[yhat < .5] = 0  
accuracy = (sum(yhat == y_holdout))/len(yhat)
print("GBLINEAR ACCURACY:")
print(accuracy)
#.7935 accuracy with nround = 5

#for creating submission only
t = xg_model.predict(d_test)
t[t >= .5] = 1
t[t < .5] = 0
id = np.arange(50000) + 1
result =  np.column_stack((id, t.astype(int)))
with open("xgboost_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")

#create holdout predictions
id = np.arange(25477) + 1
result =  np.column_stack((id, yhat.astype(int)))
with open("holdout_xgboost_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")




######TREE#######
params_tree = {
        "booster":"gbtree",
        "objective":"binary:logistic"
        }
xg_tree = xgb.train(params_tree, d_train, num_boost_round = 15)
y_tree = xg_tree.predict(d_holdout)
y_tree[y_tree >= .5] = 1
y_tree[y_tree < .5] = 0
print("GBTREE ACCURACY:")
print(sum(y_tree == y_holdout)/len(y_tree))    #.698904

#for creating submission only
t = xg_tree.predict(d_test)
t[t >= .5] = 1
t[t < .5] = 0
id = np.arange(50000) + 1
result =  np.column_stack((id, t.astype(int)))
with open("xgtree_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")


#create holdout predictions
id = np.arange(25477) + 1
result =  np.column_stack((id, y_tree.astype(int)))
with open("holdout_xgtree_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")

