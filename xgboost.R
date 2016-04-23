#install.packages("xgboost")
#install.packages("e1071")
library(xgboost)
library(spls)
library(e1071)
library(kernlab)


GetBoostProb = function(Xtrain, ytrain, Xvalid, hyperparameter, tree = TRUE) {
  if (tree) {
    gb = "gbtree"
  } else {
    gb = "gblinear"
  }
  model = xgboost(data = Xtrain, label = ytrain, max_depth = hyperparameter, nrounds = 50,
                  booster = gb)
  return(predict(model, newdata = Xvalid))
}


GetFoldID = function(n, n_fold) {
  fold_id = rep(1:n_fold, 1 + n / n_fold)
  set.seed(2)
  fold_id = sample(fold_id, size = n)
  fold_id = fold_id[1:n]
  return(fold_id)
}

GetCVPrediction = function(model, X, y, n_fold, hyperparameter, gbTree = TRUE) {
  n = length(y)
  fold_id = GetFoldID(n, n_fold)
  prob = numeric(n)
  fold_prob = numeric(n_fold)
  for (fold in 1:n_fold) {
    temp = model(X[fold_id != fold, ], y[fold_id != fold],
                 X[fold_id == fold, ], hyperparameter, tree = gbTree)
    temp[temp > .5] = 1
    temp[temp < .5] = 0
    n_fold[fold] = length(temp[temp == y[fold_id == fold]])/length(temp)
  }
  return(n_fold)
}

#vector of the CV accuracy for different values of hyperparameters
boosted_prob = numeric(10)
for (i in 1:10) {
  boosted_prob[i] = mean(
    GetCVPrediction(GetBoostProb, X_train, y_train, 5, i, gbTree = FALSE)
  )
}

#xg model with hyperparameters tuned through 5-fold CV
#nrounds gets better the higher it gets but 50 is good value
#max_depth gets better the higher it is but drops off around 9
#eta doesn't matter
xg = xgboost(data = X_train, label = y_train, eta = .3, max_depth = 9, nrounds = 50, 
             booster = "gblinear")
yhat_xg = predict(xg, newdata = X_holdout)
yhat_xg[yhat_xg >= .5] = 1
yhat_xg[yhat_xg < .5] = 0
sum(yhat_xg == y_holdout)/length(yhat_xg)

#accuracy of booster = "gbtree" is .7508%
#accuracy of booster = "gblinear" is .77

#save(xg, file = "xgboost.rda")
#dummy submission
Xtest[,1] = as.numeric(Xtest[,1])
yhat_xg = predict(xg, newdata = Xtest)
yhat_xg[yhat_xg >= .5] = 1
yhat_xg[yhat_xg < .5] = 0
submission = data.frame(id = 1:length(yhat_xg), y = yhat_xg)
colnames(submission) = c("id", "y")
write.csv(submission, file = "xgSubmission.csv", row.names = FALSE) #kaggle score: 75.8%
