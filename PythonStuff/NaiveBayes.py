from __future__ import division
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB 

X_train = np.load("Data/X_train.npy")[()]
X_holdout = np.load("Data/X_holdout.npy")[()]
y_train = np.load("Data/y_train.npy")[()]
y_holdout = np.load("Data/y_holdout.npy")[()]
X_test = np.load("Data/X_test.npy")[()]

"""
#Crashes on my mac, not enough memory
X_train = X_train.toarray()
y_train = np.toarray(y_train)
X_holdout = np.toarray(X_holdout)
y_holdout = np.toarray(y_holdout)
X_test = np.toarray(X_test)
"""

"""
#requires dense array but my computer can only handle sparse here otherwise i get segment error
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_gaussian = gaussian.predict(X_holdout)
"""
multi = MultinomialNB()
multi.fit(X_train,y_train)
y_multi = multi.predict(X_holdout)

"""
#same as gaussian
bernoulli = BernoulliNB()
bernoulli.fit(X_train, y_train)
y_bernoulli = bernoulli.predict(X_holdout)
"""
#print("Gaussian accuracy is")
#print(sum(y_gaussian == y_holdout)/len(y_gaussian))

print("Multinomial accuracy is")
print(sum(y_multi == y_holdout)/len(y_multi))     #77.2108%

#print("Bernoulli accuracy is")
#print(sum(y_bernoulli == y_holdout)/len(y_bernoulli))

#for creating submission only
t = multi.predict(X_test)
t[t >= .5] = 1
t[t < .5] = 0
id = np.arange(50000) + 1
result =  np.column_stack((id, t.astype(int)))
with open("naivebayes_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")

#create holdout predictions
id = np.arange(25477) + 1
result =  np.column_stack((id, y_multi.astype(int)))
with open("holdout_naivebayes_submission.csv", "wb") as f:
        f.write(b'Id,y\n')
        np.savetxt(f, result, fmt='%i', delimiter=",")

