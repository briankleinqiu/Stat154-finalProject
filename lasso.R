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

#TRAIN LASSO
y_train = as.integer(y_train)
y_holdout = as.integer(y_holdout)
lasso = cv.glmnet(X_train, y_train, nfolds = 10)

lasso$lambda.min #best lambda is .001135525
1 - min(lasso$cvm) #best cross-validated prediction is 82.3235%

#PREDICT ON XTEST
yhat = predict(lasso, newx = Xtest, s = "lambda.min")
yhat[yhat >= .5] = 1
yhat[yhat < .5] = 0

submission = data.frame(id = 1:length(yhat), y = yhat)
colnames(submission) = c("id", "y")
write.csv(submission, file = "lassoSubmission.csv", row.names = FALSE) #kaggle score: 75.9%

#PREDICT ON HOLDOUT
yhat = predict(lasso, newx = X_holdout, s = "lambda.min")
yhat[yhat >= .5] = 1
yhat[yhat < .5] = 0
sum(yhat == y_holdout)/length(yhat) #77.4%

#sparse logistic
sparse_log = cv.glmnet(X_train, as.factor(y_train), nfolds = 10, family = "binomial", type.measure = "class")
yhat_log = predict(sparse_log, newx = X_holdout, s = "lambda.min", type = "class")
sum(yhat_log == as.factor(y_holdout))/length(yhat) #77.16%

#find sparse indices
sparse_indices = which(coef(sparse_log, s = "lambda.1se") != 0)
sparse_indices = as.character(sparse_indices)
write(sparse_indices, "sparse_indices.txt", sep="\n")

#logistic on all variables 
temp = as.data.frame(cbind(X_train, y_train))
temp[,1001] = as.factor(temp[,1001])
colnames(temp)[1001] = "response"
logit = glm(response ~  ., data = temp , family = "binomial")
yhat_logit = predict(logit, newdata = as.data.frame(X_holdout), type = "response")
yhat_logit
yhat_logit[yhat_logit >= .5] = 1
yhat_logit[yhat_logit < .5] = 0
sum(yhat_logit == as.factor(y_holdout))/length(yhat_logit) #77%

#try naive bayes
library(e1071)
bayes = naiveBayes(x = X_train, y = as.factor(y_train))
yhat_bayes =  predict(bayes, newdata = X_holdout) 
sum(yhat_bayes == as.factor(y_holdout))/length(yhat_bayes) #68.4% on Naive Bayes

save(sparse_log, file = "sparse_logistic.rda")





