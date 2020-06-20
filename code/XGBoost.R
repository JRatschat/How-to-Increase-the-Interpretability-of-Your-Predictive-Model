# Packages
library(readr)
library(dplyr)
library(data.table)
library(xgboost)
library(caret)
library(ggplot2)
library(cowplot)
library(SHAPforxgboost)
library(pROC)

# Load data
churn <- read.csv("data/datasets_156197_358170_Churn_Modelling.csv")

# Explore data
head(churn)
str(churn)
summary(churn)

# Delete "unneccessary" columns
churn <- churn %>%
  dplyr::select(-RowNumber, -CustomerId, -Surname)

# Transform all character variables into factor variables
churn <- churn %>% 
  dplyr::mutate(Geography = as.factor(Geography),
                Gender = as.factor(Gender))

# Transform integer variable NumOfProducts to factor
churn <- churn %>% 
  dplyr::mutate(NumOfProducts = as.factor(NumOfProducts))

# Create one-hot coded matrix
churn <-as.data.frame(model.matrix(~.-1, data = churn))

# Randomly select 80% of the observations without replacement 
set.seed(20)
train_id <- sample(1:nrow(churn), size = floor(0.8 * nrow(churn)), replace=FALSE) 

# Split in train and validation (80/20)
churn_first_round_training <- churn[train_id,]
churn_test <- churn[-train_id,]

# Randomly select 80% of the observations without replacement
set.seed(20)
train_id <- sample(1:nrow(churn_first_round_training), size = floor(0.8 * nrow(churn_first_round_training)), replace=FALSE) 

# Split in Training and Validation (80/20)
churn_train <- churn_first_round_training[train_id,]
churn_valid <- churn_first_round_training[-train_id,]

# Returns the NA object unchanged, if not changed, NA would be dropped
options(na.action='na.pass')

# Prepare matrix for XGBoost algorithm
training_matrix <-model.matrix(Exited ~.-1, data = churn_train)
validation_matrix <-model.matrix(Exited ~.-1, data = churn_valid)
test_matrix <-model.matrix(Exited ~.-1, data = churn_test)
dtrain <- xgb.DMatrix(data = training_matrix, label = churn_train$Exited) 
dvalid <- xgb.DMatrix(data = validation_matrix, label = churn_valid$Exited)
dtest <- xgb.DMatrix(data = test_matrix, label = churn_test$Exited)

# Base XGBoost model
set.seed(20)
params <- list(booster = "gbtree", 
               objective = "binary:logistic")
xgb_base <- xgb.train (params = params,
                       data = dtrain,
                       nrounds =1000,
                       print_every_n = 10,
                       eval_metric = "auc",
                       eval_metric = "error",
                       early_stopping_rounds = 50,
                       watchlist = list(train= dtrain, val= dvalid))

# Make prediction on dvalid
churn_valid$pred_churn_base <- predict(xgb_base, dvalid)
churn_valid$pred_churn_factor_base <- factor(ifelse(churn_valid$pred_churn_base > 0.5, 1, 0), 
                                               labels=c("Not Churned","Churned"))

# Check accuracy with the confusion matrix
confusionMatrix(churn_valid$pred_churn_factor_base, 
                factor(churn_valid$Exited ,
                       labels=c("Not Churned", "Churned")),
                positive = "Churned", 
                dnn = c("Prediction", "Actual Data"))

# Make prediction on dtest
churn_test$pred_churn_base <- predict(xgb_base, dtest)
churn_test$pred_churn_factor_base <- factor(ifelse(churn_test$pred_churn_base > 0.5, 1, 0), 
                                             labels=c("Not Churned","Churned"))

# Check accuracy with the confusion matrix
confusionMatrix(churn_test$pred_churn_factor_base, 
                factor(churn_test$Exited ,
                       labels=c("Not Churned", "Churned")),
                positive = "Churned", 
                dnn = c("Prediction", "Actual Data"))

# ROC
roc_test <- roc(churn_test$Exited, churn_test$pred_churn_base, percent=TRUE, plot=TRUE, print.auc=TRUE,grid=TRUE)
roc_test

# Set theme for plots
mytheme <- theme(
  ## plotregion
  panel.background = element_rect(fill = "white",
                                  colour = "white"),
  panel.border = element_blank(),
  panel.grid.major = element_line(size = 0.5, 
                                  linetype = 'solid',
                                  colour = "lightgrey"),
  panel.grid.minor = element_blank(),
  panel.spacing = unit(0.25, "lines"),
  ## axis line
  axis.line = element_line(colour = "black",
                           size = 0.5, 
                           linetype = "solid"),
  ##background of fill
  strip.background = element_rect(fill="#f2f2f2") +
    scale_fill_discrete())

# Feature importance Gain
importance_matrix <- xgb.importance(model = xgb_base)
g1 <- xgb.ggplot.importance(as.data.table(importance_matrix), measure = "Gain", n_clusters = 1) +
  labs(title = "") +
  xlab("") + 
  ylab("Gain") +
  mytheme +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("#008AE7", "#FF0056"))

# Feature performance and relevance
importance_matrix <- xgb.importance(model = xgb_base)
g2 <- xgb.ggplot.importance(as.data.table(importance_matrix), measure = "Frequency", n_clusters = 1) +
  labs(title = "") +
  xlab("") + 
  ylab("Frequency") +
  mytheme +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("#008AE7", "#FF0056"))

# Feature performance and relevance
importance_matrix <- xgb.importance(model = xgb_base)
g3 <- xgb.ggplot.importance(as.data.table(importance_matrix), measure = "Cover", n_clusters = 1) +
  labs(title = "") +
  xlab("") + 
  ylab("Cover") +
  mytheme +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("#008AE7", "#FF0056"))

plot_grid(g1, g2, g3, align = "v", nrow = 3, rel_heights = c(1/3, 1/3, 1/3))

# SHAP preparation
set.seed(20)
shap_values <- shap.values(xgb_base, X_train = training_matrix)
shap_values$mean_shap_score

# Prepare long-format data
set.seed(20)
shap_long <- shap.prep(xgb_model = xgb_base, X_train = training_matrix)
# is the same as: using given shap_contrib
shap_long <- shap.prep(shap_contrib = shap_values$shap_score, X_train = training_matrix)

# SHAP Dependence plot: Age
shap_int <- shap.prep.interaction(xgb_mod = xgb_base, X_train = training_matrix)

shap.plot.dependence(data_long = shap_long, x = "Age", y = 'Age',  smooth = FALSE) + 
  mytheme +
  scale_color_gradient(low = "#008AE7", high = "#FF0056",
                       breaks=c(0,1), labels=c("0","1"),
                       guide = guide_colorbar(barwidth = 2, barheight = 0.6)) +
  labs(color = "GenderMale  ") +
  geom_hline(yintercept=0, colour = "darkgrey")

# SHAP Dependence and Interaction plot: Age
g1 <- shap.plot.dependence(data_long = shap_long, x = "Age", y = 'Age',  smooth = FALSE, color_feature = "GenderMale") + 
  mytheme +
  scale_color_gradient(low = "#008AE7", high = "#FF0056",
                       breaks=c(0,1), labels=c("0","1"),
                       guide = guide_colorbar(barwidth = 2, barheight = 0.6)) +
  labs(color = "GenderMale  ") +
  geom_hline(yintercept=0, colour = "darkgrey")

g2 <- shap.plot.dependence(data_long = shap_long, data_int = shap_int, x= "Age", y = "GenderMale", smooth = FALSE, color_feature = "GenderMale") +
  mytheme +
  scale_color_gradient(low = "#008AE7", high = "#FF0056",
                       breaks=c(0,1), labels=c("0","1"),
                       guide = guide_colorbar(barwidth = 2, barheight = 0.6)) +
  labs(color = "GenderMale  ") +
  geom_hline(yintercept=0, colour = "darkgrey")

gridExtra::grid.arrange(g1, g2, ncol = 2)