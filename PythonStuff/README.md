
INSTRUCTIONS

1. Download the MaskedDataRaw.csv file and vocab.pkl file as provided by Kaggle into the Data folder 
2. Run 'python ProcessData.py' from this directory to use ProcessData.py provided by Hoang to create the large 1578627 x 5000 matrix pyMatrix and the 5000 long word vector pyVocab, all stored in Data. WARNING: Takes a long time
3. Run 'python MakeData.py' from this directory to generate training sets and holdout sets X\_Train, y\_Train, X\_Holdout, y\_Holdout in the Data folder. X\_Train should be 1503150 x 5000 matrix y\_Train should be size 1503150, X\_Holdout is 25477 x 5000 and y\_Holdout is size 25477


MODELS
Each of these files creates a model, a prediction on the holdout set, and a prediction on the test set used for submission as a csv file:

- SparseLogistic.py creates a sparse logistic model (logistic model with L1 penalty and C parameter tuned to .9), accuracy is 79.6248% 
- XGboost.py creates 2 xgboost models using a linear method and tree, some parameters tuned using XGBoost\_CV.py, accuracy is 79.35% for linear and 69.89% for tree
- NaiveBayes.py creates the Multinomial model, can't get Gaussian/Bernoulli to work on my laptop. 77.2108%
- SVM.py creates a linear SVM model with L1 penalty, takes forever. 79.56% on holdout with L2 penalty and identical to logistic on Kaggle, .79585 with L1 penalty
- kernelSVM.py creates a kernel SVM model, takes hours so run with caution.


MISC:

- MakeData.py and ProcessData.py create the datasets per the instructions
- Not used: PCA.py attempts to run PCA on the dataset, however PCA functions won't work on sparse scipy, so use TruncatedSVD instead




TODO: 

- Build SVM, CV/tune C for .1 to 1 and 1 to 10 for linear if time, build kernel method #1 is best parameter for C
- tune C for logistic #.9 is best
- Build Random Forest? Might not be necessary since xgboost tree
- Build K nearest neighbours classifier
- Build adaboost
- Remove correlated coefficients/build sparse datasets and run again, particularly on Naive Bayes #doesn't improve when using sparse datasets from logistic or svm
- If time consider bagging for ensembling especially for Naive Bayes?
- Implement majority vote


