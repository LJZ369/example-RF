# Here, we provide an example where 30 points were randomly selected from 
# each of the five marine regions. The Random Forest model was then employed for 
# model training, which included steps such as feature selection, 
# hyperparameter optimization, and model evaluation.
# If there are any issues with the code execution, please contact lin_jiezhang@126.com.
library(tidyverse)
library(mlr3verse)
library(tidymodels)

# read data----
mps.data.calibrated <- read.csv('mps_all_exp.csv')
# Normalized MP concentrations  
mps_task <- mps.data.calibrated[,-c(1:2)] |> mutate(conc = log10(conc+1))
# Training set/test set  8/2
set.seed(123)
mps_task_split <- initial_split(mps_task, prop = 0.8, strata = 'sea')
mps_task_train <- training(mps_task_split)
mps_task_test  <-  testing(mps_task_split)

# Parallelize processing, 
future::plan("multisession", workers = 20)
# RF - reg 
task_for_RF_train <- as_task_regr(mps_task_train , target = 'conc')
# Set up stratified sampling
task_for_RF_train$set_col_roles('sea', c('stratum'))
# Random Forest model
lrn_RF <- lrn('regr.ranger')

set.seed(124) 
# Forward sequential selection, rmse, 10 cv
feature_select_for_RF = fselect(
  fselector =fs("sequential"),
  task =  task_for_RF_train,
  learner = lrn_RF,
  resampling = rsmp("cv", folds = 10),
  measure = msr('regr.rmse')
)

task_for_RF_train$select(feature_select_for_RF$result_feature_set)
max.mtry = length(feature_select_for_RF$result_feature_set)
# Hyperparametric optimization, 10-grid
grid_search_10 = tnr("grid_search", resolution = 10)
set.seed(125)
hyperparameter_for_RF = ti(
    task = task_for_RF_train,
    learner = lrn_RF,
    resampling = rsmp("cv", folds = 10),
    measures = msr('regr.rmse'),
    search_space = ps(mtry  = p_int(lower = 1, upper = max.mtry), 
      num.trees = p_int(lower = 100, upper =  1900)),
    terminator = trm("none")
  )
grid_search_10$optimize(hyperparameter_for_RF)

# Random forest model on the entire training set, set optimal parameters, select features
lrn_RF_tune <-  lrn('regr.ranger')
lrn_RF_tune$param_set$values <- hyperparameter_for_RF$result_learner_param_vals
lrn_RF_tune$train(task_for_RF_train)

# Evaluation of Random Forest 
prediction_RF_train = lrn_RF_tune$predict_newdata(mps_task_train)
prediction_RF_test = lrn_RF_tune$predict_newdata(mps_task_test)
rf_train = prediction_RF_train$score(msrs(c('regr.rsq','regr.rmse')))
rf_test = prediction_RF_test$score(msrs(c('regr.rsq','regr.rmse')))

# Output results
# 10-fold cross-validation results on the training set
# RMSE
print(round(hyperparameter_for_RF$result$regr.rmse,3))
# Results on the entire training set, RMSE, R2
print(round(rf_train,3))
# Results on the entire test set, RMSE, R2
print(round(rf_test,3))