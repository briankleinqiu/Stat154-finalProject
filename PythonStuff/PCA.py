from __future__ import division
import numpy as np
from sklearn.decomposition import TruncatedSVD

#Function to test PCA in Python and find how many components to keep. 
#Edit: despite the name, neither PCA nor RandomizedPCA will work on a scipy sparse matrix so use TruncatedSVD instead which is similar

X_train = np.load("Data/X_train.npy")[()]
X_holdout = np.load("Data/X_holdout.npy")[()]

pca = TruncatedSVD(n_components = 100)
pca.fit(X_train)
print("Explained Variance")
explained = pca.explained_variance_ratio_
print(explained[::9])
print(explained.sum())

np.save("Data/SVD_X_train", pca.components_)

pca_hold = TruncatedSVD(n_components = 100)
pca_hold.fit(X_holdout)
print("Explained Variance of holdout")
exp_hold = pca_hold.explained_variance_ratio_
print(exp_hold[::9])
print(exp_hold.sum())

np.save("Data/SVD_X_test", pca_hold.components_)
