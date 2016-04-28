from __future__ import division
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation

#Takes a couple minutes to run the program


X_train = np.load("Data/X_train.npy")[()]
X_holdout = np.load("Data/X_holdout.npy")[()]
y_train = np.load("Data/y_train.npy")[()]
y_holdout = np.load("Data/y_holdout.npy")[()]
X_test = np.load("Data/X_test.npy")[()]

knn = KNeighborsClassifier(
        n_neighbors = 5,
        leaf_size = 30,
        #metric = "minkowsi", #distance method, minkowski or euclidean
        #p = 2 #parameter for minkowski, 1 = l1 manhattan, 2 = l2 euclidean
        )

knn.fit(X_train, y_train)
y_knn = knn.predict(X_holdout)
accuracy = (sum(y_knn == y_holdout))/len(y_knn)
print(accuracy)

#for creating submission only
t = knn.predict(X_test)
id = np.arange(50000) + 1
result =  np.column_stack((id, t.astype(int)))
with open("knn_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")


#create holdout predictions
id = np.arange(25477) + 1
result =  np.column_stack((id, y_knn.astype(int)))
with open("holdout_knn_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")
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
