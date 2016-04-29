from __future__ import division
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


X_train = np.load("../Data/X_train.npy")[()]
X_holdout = np.load("../Data/X_holdout.npy")[()]
y_train = np.load("../Data/y_train.npy")[()]
y_holdout = np.load("../Data/y_holdout.npy")[()]
X_test = np.load("../Data/X_test.npy")[()]

dt = DecisionTreeClassifier(
        criterion = "gini",
        max_depth = 2
        #min_samples_leaf = 50
        )

ada = AdaBoostClassifier(
        base_estimator = dt,
        n_estimators = 10,
        learning_rate = 1
        )

ada.fit(X_train, y_train)
y_ada = ada.predict(X_holdout)
y_ada[y_ada >= .5] = 1
y_ada[y_ada < .5]= 0
print(sum(y_ada == y_holdout)/len(y_ada))
#66.0988


#for creating submission only
t = ada.predict(X_test)
id = np.arange(50000) + 1
result =  np.column_stack((id, t.astype(int)))
with open("ada_decisiontree_submission2.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")


#create holdout predictions
id = np.arange(25477) + 1
result =  np.column_stack((id, y_ada.astype(int)))
with open("ada_holdout_decisiontree_submission2.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")
