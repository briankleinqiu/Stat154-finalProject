from __future__ import division
import numpy as np
from sklearn.linear_model import Perceptron 
from sklearn import cross_validation

X_train = np.load("Data/X_train.npy")[()]
X_holdout = np.load("Data/X_holdout.npy")[()]
y_train = np.load("Data/y_train.npy")[()]
y_holdout = np.load("Data/y_holdout.npy")[()]
X_test = np.load("Data/X_test.npy")[()]

p = Perceptron(
        penalty = "l1",
        alpha = .0001,
        eta0 = 1
        )

p.fit(X_train, y_train)
y_p = p.predict(X_holdout)
p_accuracy = sum(y_p == y_holdout)/len(y_p)
print("Accuracy:")
print(p_accuracy) #.6829

#for creating submission only
t = p.predict(X_test)
t[t >= .5] = 1
t[t < .5] = 0
id = np.arange(50000) + 1
result =  np.column_stack((id, t.astype(int)))
with open("perceptron_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")

#create holdout predictions
id = np.arange(25477) + 1
result =  np.column_stack((id, y_p.astype(int)))
with open("holdout_perceptron_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")
"""
print("Cross Validation")
values = [x/10 for x in range(1, 11)]
#values = values[::2]
for i in values:
    print(i)
    p = Perceptron(
        penalty = "l1",
        alpha = .0001,
        eta0 = i 
        )                  
    scores = cross_validation.cross_val_score(p, X_train, y_train, cv = 5)
    print(scores.mean(), scores.std()*2)

#for penalty = None is .6929
# for penalty = l1 is .6947
#for penalty = l2 is .66
#eta0 = .1 is highest
"""
