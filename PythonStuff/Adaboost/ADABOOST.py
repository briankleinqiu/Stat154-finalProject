from __future__ import division
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier

X_train = np.load("../Data/X_train.npy")[()]
X_holdout = np.load("../Data/X_holdout.npy")[()]
y_train = np.load("../Data/y_train.npy")[()]
y_holdout = np.load("../Data/y_holdout.npy")[()]
X_test = np.load("../Data/X_test.npy")[()]

y_train[y_train == 0] = -1
y_holdout[y_holdout == 0] = -1

def geterror(model, weights, predictions):
    #recode false to 1, true to 0
    compare = (y_train == predictions)
    compare = (compare - 1) * - 1
    temp = weights.dot(compare)
    error = np.sum(temp)
    return(error/np.sum(weights))

def reweight(weights, alpha, predictions):
    compare = (y_train == predictions)
    compare = (compare - 1) * - 1
    temp = alpha * compare
    return(weights * np.exp(temp))



n = X_train.shape[0] #number of training samples
alphas = []
w = np.repeat(1/n, n) #weights vector initialized to 1/n

#fit sparse on it
logit = LogisticRegression(penalty = 'l1', C = 0.9)
logit.fit(X_train, y_train)
p = logit.predict(X_train)
print(sum(p == y_train)/len(p))
error = geterror(logit, w, p)
alphas.append(np.log((1 - error)/error))

#reweight
w = reweight(w, alphas[0], p)
w = w/np.sum(w)
print(np.unique(w))



#fit random forest on it
rf = RandomForestClassifier(
        n_estimators = 10,
        criterion = "gini",
        max_features = "auto",
        min_samples_leaf = 50
        )
rf.fit(X_train, y_train, sample_weight = w)
p = rf.predict(X_train)
print(sum(p == y_train)/len(p))

#reweight
error = geterror(logit, w, p)
alphas.append(np.log((1 - error)/error))
w = reweight(w, alphas[1], p)


final_pred = alphas[0] * logit.predict(X_holdout) + alphas[1] * rf.predict(X_holdout)
final_pred = np.sign(final_pred)
print(sum(final_pred == y_holdout)/len(final_pred))

