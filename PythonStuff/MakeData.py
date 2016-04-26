from __future__ import division
import numpy as np

#Create X_test, y_test, X_train, y_train
X = np.load("Data/pyMatrix.npz")['X'][()]
y = np.load("Data/pyMatrix.npz")['y'][()]

y_test = y[0:50000]
y = y[50000:]
X_test = X[0:50000,]
X = X[50000:,]

#Holdout set is 25477, roughly 1/61 of the data
holdout_indices = np.random.choice(np.arange(1528627), size = 25477, replace = False)
temp = np.repeat(True, 1528627)
temp[holdout_indices] = False
train_indices = np.arange(1528627)[temp]
X_train = X[train_indices,]
X_holdout = X[holdout_indices,]
y_train = y[train_indices,]
y_holdout = y[holdout_indices,]

print("y_train is ", len(y_train), " and y_holdout is ", len(y_holdout))
print("X_train is ", X_train.shape, " X_holdout is ", X_holdout.shape)  #save files and print to see if X,y dimensions correct

np.save("Data/X_train", X_train) 
np.save("Data/y_train", y_train)
np.save("Data/X_holdout", X_holdout)
np.save("Data/y_holdout", y_holdout)
np.save("Data/X_test", X_test)
