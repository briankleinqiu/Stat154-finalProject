from __future__ import division
import numpy as np
from sklearn import svm

X_train = np.load("Data/X_train.npy")[()]
X_holdout = np.load("Data/X_holdout.npy")[()]
y_train = np.load("Data/y_train.npy")[()]
y_holdout = np.load("Data/y_holdout.npy")[()]
X_test = np.load("Data/X_test.npy")[()]

kernel = svm.SVC(kernel = "rbf")
kernel.fit(X_train, y_train)
y_kernel = kernel.predict(X_holdout)
kernel_accuracy = sum(y_kernel == y_holdout)/len(y_kernel)
print("Kernel:")
print(kernel_accuracy) 

#for creating submission only
t = kernel.predict(X_test)
t[t >= .5] = 1
t[t < .5] = 0
id = np.arange(50000) + 1
result =  np.column_stack((id, t.astype(int)))
with open("kernelsvm_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")

#create holdout predictions
id = np.arange(25477) + 1
result =  np.column_stack((id, y_kernel.astype(int)))
with open("holdout_kernelsvm_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")

