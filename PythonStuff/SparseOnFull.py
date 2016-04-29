from __future__ import division
import numpy as np
import sklearn.linear_model
from sklearn import cross_validation
from scipy.sparse import vstack

X_train = np.load("Data/X_train.npy")[()]
X_holdout = np.load("Data/X_holdout.npy")[()]
y_train = np.load("Data/y_train.npy")[()]
y_holdout = np.load("Data/y_holdout.npy")[()]
X_test = np.load("Data/X_test.npy")[()]

X_train = vstack((X_train, X_holdout))
y_train = np.concatenate((y_train, y_holdout))

print(X_train.shape)
print(y_train.shape)


logit = sklearn.linear_model.LogisticRegression(penalty = 'l1', C = 0.9)
logit.fit(X_train, y_train)
yhat_logit = logit.predict(X_holdout)
accuracy = (sum(yhat_logit == y_holdout))/len(yhat_logit)

print(accuracy)

#for creating submission only
t = logit.predict(X_test)
id = np.arange(50000) + 1
result =  np.column_stack((id, t.astype(int)))
with open("logistic_full_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")

"""
#create holdout predictions
id = np.arange(25477) + 1
result =  np.column_stack((id, yhat_logit.astype(int)))
with open("holdout_logistic_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")
"""
"""
print("Cross Validation")
values = [x/10 for x in range(10, 15)]
values = values[::2]
for i in values:
    print(i)
    temp = sklearn.linear_model.LogisticRegression(penalty = 'l1', C = i)
    scores = cross_validation.cross_val_score(temp, X_train, y_train, cv = 5)
    print(scores.mean(), scores.std()*2)
#79.3386 for C = 9, 79.3375 for C = 1
"""
