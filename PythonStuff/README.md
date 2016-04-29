
INSTRUCTIONS

1. Download the MaskedDataRaw.csv file and vocab.pkl file as provided by Kaggle into the Data folder 
2. Run 'python ProcessData.py' from this directory to use ProcessData.py provided by Hoang to create the large 1578627 x 5000 matrix pyMatrix and the 5000 long word vector pyVocab, all stored in Data. WARNING: Takes a long time
3. Run 'python MakeData.py' from this directory to generate training sets and holdout sets X\_Train, y\_Train, X\_Holdout, y\_Holdout in the Data folder. X\_Train should be 1503150 x 5000 matrix y\_Train should be size 1503150, X\_Holdout is 25477 x 5000 and y\_Holdout is size 25477


MODELS
Each of these files creates a model, a prediction on the holdout set, and a prediction on the test set used for submission as a csv file:

- SparseLogistic.py creates a sparse logistic model (logistic model with L1 penalty and C parameter tuned to .9), accuracy is 79.6248% 
- XGboost.py creates 2 xgboost models using a linear method and tree, some parameters tuned using XGBoost\_CV.py, accuracy is 79.546% for linear and 69.89% for tree
- NaiveBayes.py creates the Multinomial model, can't get Gaussian/Bernoulli to work on my laptop. 77.2108%. However multinomial is most suited for discrete count data anyway.
- SVM.py creates a linear SVM model with L1 penalty, takes forever. 79.56% on holdout with L2 penalty and identical to logistic on Kaggle, .79585 with L1 penalty
- kernelSVM.py creates a kernel SVM model with Gaussian rbf kernel, is O(n^2) so train on a smaller subset. 65.577% with rbf, 50.18644 with poly 
- KNN.py: doesn't work due to errors with scipy sparse matrix 
- RandomForest.py creates random forest with 'gini' and min\_leaf set to 100. Ran on whole dataset of n\_estimators = 10. Gives 74.632%
- DecisionTree.py, too slow to finish in time for deadline
- Perceptron.py creates perceptron separation, best with l1 penalty and gives 68.29


ENSEMBLES
- Adaboost folder contains adaboost files, however could not get the sklearn version to run due to computational limitations, so ADABOOST.py is a rough attempt at a manual implementation.
    * Currently implements sparse logistic, then random forest, then naive bayes
- Stack.py and Stacking.ipynb are an attempt at stacking through logistic regression, based on code from Eric Chio and Emanuele Olivetti
- MajorityVote.py implements a majority vote of all the classifiers, returns 79.636% on the holdout 


MISC:

- MakeData.py and ProcessData.py create the datasets per the instructions
- Not used: PCA.py attempts to run PCA on the dataset, however PCA functions won't work on sparse scipy, so use TruncatedSVD instead




TODO: 

- Build SVM, CV/tune C for .1 to 1 and 1 to 10 for linear if time
    * 1 is best parameter for C
- Build Kernel SVM
    * On 50,000 observations using Gaussian kernel and poly kernel
- tune C for logistic 
    * .9 is best
- Build Random Forest? Might not be necessary since xgboost tree 
    * minleaf 50 or 100 makes almost no difference 
- Tune random forest thru CV for n\_estimators, min\_samples\_leaf, max\_features
    * might not be feasible computationally
- Build K nearest neighbours classifier and tune parameters
    * can't build in python due to scipy sparse difficulties with dot product of itself here
- Build adaboost
- Remove correlated coefficients/build sparse datasets from sparse logistic and sparse linear svm run again, particularly on Naive Bayes 
    * Naive Bayes: doesn't improve when using sparse datasets from logistic or svm
    * KNN: 
- If time consider bagging for ensembling especially for Naive Bayes?
- Implement majority vote
- Perceptron/LDA
- Neural Network
