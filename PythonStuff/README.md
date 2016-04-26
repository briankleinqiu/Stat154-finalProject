
INSTRUCTIONS

1. Download the MaskedDataRaw.csv file and vocab.pkl file as provided by Kaggle into the Data folder 
2. Run 'python ProcessData.py' from this directory to use ProcessData.py provided by Hoang to create the large 1578627 x 5000 matrix pyMatrix and the 5000 long word vector pyVocab, all stored in Data. WARNING: Takes a long time
3. Run 'python MakeData.py' from this directory to generate training sets and holdout sets X\_Train, y\_Train, X\_Holdout, y\_Holdout in the Data folder. X\_Train should be 1503150 x 5000 matrix y\_Train should be size 1503150, X\_Holdout is 25477 x 5000 and y\_Holdout is size 25477

Each of these files creates a model and a prediction on the test set used for submission as a csv file:

- SparseLogistic.py creates a sparse logistic model (logistic model with L1 penalty), accuracy is 79.609% 
- XGboost.py creates 2 xgboost models using a linear method and tree, some parameters tuned using XGBoost\_CV.py, accuracy is 79.35% for linear and 69.89% for tree
- NaiveBayes.py creates the Multinomial model, can't get Gaussian/Bernoulli to work on my laptop. 77.2108%


TODO: 

- Build SVM 
- Build Random Forest? Might not be necessary since xgboost tree
- Implement majority vote
- Build adaboost
- Remove correlated coefficients and run again, particularly on Naive Bayes

