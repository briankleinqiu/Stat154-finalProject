from __future__ import division
import numpy as np
from sklearn.ensemble import RandomForestClassifier

X_train = np.load("Data/X_train.npy")[()]
X_holdout = np.load("Data/X_holdout.npy")[()]
y_train = np.load("Data/y_train.npy")[()]
y_holdout = np.load("Data/y_holdout.npy")[()]
X_test = np.load("Data/X_test.npy")[()]

rf = RandomForestClassifier(
        n_estimators = 10,
        criterion = "gini",
        max_features = "auto",
        min_samples_leaf = 50
        )

rf.fit(X_train, y_train)
y_rf = rf.predict(X_holdout)
y_rf[y_rf >= .5] = 1
y_rf[y_rf < .5] = 0
print(sum(y_rf == y_holdout)/len(y_rf)) 
#74.632 with n_estimators = 10, min_samples_leaf = 50
#




#for creating submission only
t = rf.predict(X_test)
id = np.arange(50000) + 1
result =  np.column_stack((id, t.astype(int)))
with open("randomforest_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")


#create holdout predictions
id = np.arange(25477) + 1
result =  np.column_stack((id, y_rf.astype(int)))
with open("holdout_randomforest_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")
"""
print("Cross Validation")
values = [x for x in range(, 100)]
values = values[::20]
for i in values:
    print(i)
    temp = RandomForestClassifier(
           n_estimators = 10,
           criterion = "gini",
           max_features = "auto",
           min_samples_leaf = i
           )
    scores = cross_validation.cross_val_score(temp, X_train, y_train, cv = 5)
    print(scores.mean(), scores.std()*2)
"""
