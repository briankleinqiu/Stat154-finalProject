from __future__ import division
import numpy as np
import sklearn.linear_model

#Takes a minute to run the program


X_train = np.load("Data/X_train.npy")[()]
X_holdout = np.load("Data/X_holdout.npy")[()]
y_train = np.load("Data/y_train.npy")[()]
y_holdout = np.load("Data/y_holdout.npy")[()]

logit = sklearn.linear_model.LogisticRegression(penalty = 'l1')
logit.fit(X_train, y_train)
yhat_logit = logit.predict(X_holdout)
accuracy = (sum(yhat_logit == y_holdout))/len(yhat_logit)

print(accuracy)
#.793225
"""
for creating submission only
t = logit.predict(X_test)
id = np.arange(50000) + 1
result =  np.column_stack((id, yhat.astype(int)))
with open("logistic_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
            np.savetxt(f, result, fmt='%i', delimiter=",")
"""


