# Packages
library(readr)
library(dplyr)
library(data.table)
library(caret)
library(ggplot2)
library(cowplot)
library(pROC)
library(jtools)

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

summary(churn)

# Randomly select 80% of the observations without replacement 
set.seed(20)
train_id <- sample(1:nrow(churn), size = floor(0.8 * nrow(churn)), replace=FALSE) 

# Split in train and validation (80/20)
churn_train <- churn[train_id,]
churn_test <- churn[-train_id,]

# Logit model
model_logit <- glm(Exited ~., data = churn_train, family = "binomial")
summ(model_logit)

# Make prediction
churn_test$pred_logit <- predict(model_logit, churn_test, type = "response")
churn_test$pred_logit_factor <- factor(ifelse(churn_test$pred_logit> 0.5, 1, 0), labels=c("Not Churned","Churned"))

#Check accuracy with the confusion matrix
confusionMatrix(churn_test$pred_logit_factor, factor(churn_test$Exited ,labels=c("Not Churned","Churned")), 
                positive="Churned",dnn = c("Prediction", "Actual Data"))

# ROC
roc_test_logit <- roc(churn_test$Exited, churn_test$pred_logit, percent=TRUE, plot=TRUE, print.auc=TRUE,grid=TRUE)
roc_test_logit
