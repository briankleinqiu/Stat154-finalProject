from __future__ import division
import numpy as np
from sklearn.linear_model import Perceptron 
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier

X_train = np.load("../Data/X_train.npy")[()]
X_holdout = np.load("../Data/X_holdout.npy")[()]
y_train = np.load("../Data/y_train.npy")[()]
y_holdout = np.load("../Data/y_holdout.npy")[()]
X_test = np.load("../Data/X_test.npy")[()]

s = SGDClassifier(
        penalty = "l1",
        )

p = Perceptron(
        penalty = "l1",
        alpha = .0001,
        eta0 = 1
        )

ada = AdaBoostClassifier(
        base_estimator = s,
        n_estimators = 1,
        learning_rate = 1,
        algorithm = "SAMME"
        )

ada.fit(X_train, y_train)
y_ada = ada.predict(X_holdout)
#y_ada[y_ada >= .5] = 1
#y_ada[y_ada < .5]= 0
print(sum(y_ada == y_holdout)/len(y_ada))
print(y_ada[0:50])

#for creating submission only
t = ada.predict(X_test)
id = np.arange(50000) + 1
result =  np.column_stack((id, t.astype(int)))
with open("ada_perceptron_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")


#create holdout predictions
id = np.arange(25477) + 1
result =  np.column_stack((id, y_ada.astype(int)))
with open("ada_holdout_perceptron_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")




























