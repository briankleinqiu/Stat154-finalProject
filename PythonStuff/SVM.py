from __future__ import division
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn import feature_selection

"""l1 penalty seems to be faster and predict slightly better so possible to cross validate with that"""


X_train = np.load("Data/X_train.npy")[()]
X_holdout = np.load("Data/X_holdout.npy")[()]
y_train = np.load("Data/y_train.npy")[()]
y_holdout = np.load("Data/y_holdout.npy")[()]
X_test = np.load("Data/X_test.npy")[()]

linear = svm.LinearSVC(penalty = 'l1', dual = False, C = 1)
linear.fit(X_train, y_train)
y_linear = linear.predict(X_holdout)
linear_accuracy = sum(y_linear == y_holdout)/len(y_linear)
print("Linear:")
print(linear_accuracy) #.7956 for L2, .79585 for L1 penalty

#for creating submission only
t = linear.predict(X_test)
t[t >= .5] = 1
t[t < .5] = 0
id = np.arange(50000) + 1
result =  np.column_stack((id, t.astype(int)))
with open("linearsvm_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")

#create holdout predictions
id = np.arange(25477) + 1
result =  np.column_stack((id, y_linear.astype(int)))
with open("holdout_linearsvm_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")

#create sparse dataset
indices = feature_selection.SelectFromModel(linear, prefit = True)
sparse_X_train = indices.transform(X_train)
sparse_X_test = indices.transform(X_test)

print("X_train, X_test indices")
np.save("Data/sparse_X_train", sparse_X_train)
np.save("Data/sparse_X_test", sparse_X_test)

"""
print("Cross Validation")
for i in np.arange(9) + 1:
    temp = svm.LinearSVC(penalty = 'l1', dual = False, C = i)
    scores = cross_validation.cross_val_score(temp, X_train, y_train, cv = 5)
    print(scores.mean(), scores.std()*2)

#C = 1 is best
"""
