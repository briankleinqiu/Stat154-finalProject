library(randomForest)
library(rpart)
library(glmnet)

#Data
################
#Load in data
load("TrainTest.RData")
X = as.matrix(X) #X is the training matrix of 50,000 tweets and the columns are the word counts
Xtest = as.matrix(Xtest) #Xtest is the testing matrix

#Create holdout set of 5% and training set of 95%
set.seed(415250328)
holdout_indices = sample(1:50000, 2500, replace = FALSE )
temp = rep(TRUE, 50000)
temp[holdout_indices] = FALSE
train_indices = (1:50000)[temp]

X_holdout = X[holdout_indices,]
X_train = X[train_indices,]
y_train = y[train_indices]
y_holdout = y[holdout_indices]
##################

#lasso with full dataset
lasso = cv.glmnet(X, y, nfolds = 10)

lasso$lambda.min #best lambda is .001150176
1 - min(lasso$cvm) #best prediction is 82.3235%

yhat = predict(lasso, newx = Xtest, s = "lambda.min")
yhat[yhat >= .5] = 1
yhat[yhat < .5] = 0

submission = data.frame(id = 1:length(yhat), y = yhat)
colnames(submission) = c("id", "y")
write.csv(submission, file = "lassoSubmission.csv", row.names = FALSE) #kaggle score: 75.9%

#lasso's prediction on holdout is .77, much closer to Kaggle score
yhat = predict(lasso, newx = X_holdout, s = "lambda.min")
yhat[yhat >= .5] = 1
yhat[yhat < .5] = 0
sum(yhat == y_holdout)/length(yhat)

#sparse logistic
y_train = as.factor(y_train)
y_holdout = as.factor(y_holdout)
sparse_log = cv.glmnet(X_train, y_train, nfolds = 10, family = "binomial", type.measure = "class")
yhat = predict(sparse_log, newx = X_holdout, s = "lambda.min", type = "class")
sum(yhat == y_holdout)/length(yhat)

#find sparse indices
sparse_indices = which(coef(sparse_log, s = "lambda.1se") != 0)
sparse_indices = as.character(sparse_indices)
write(sparse_indices, "sparse_indices.txt", sep="\n")


