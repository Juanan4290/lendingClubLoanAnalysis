##### 1. Libraries #####

library(caTools)
library(dplyr)
library(stringr)
library(ROCR)


##### 2. Read Data #####

loans <- readRDS("/media/juanan/DATA/loan_data_analysis/data/loansFirstApproach.rds")

head(loans)
str(loans)
summary(loans)


##### 3. Train / Test split #####

set.seed(4290)
sampleSplit = sample.split(loans$loan_status, SplitRatio = .75)
loansTrain = subset(loans, sampleSplit == TRUE)
loansTest = subset(loans, sampleSplit == FALSE)


##### 5. Model and feature selection #####

model=glm(loan_status~., data=loansTrain,family=binomial(link="logit"))
summary(model)

finalModel <- step(model,direction="both",trace=1)
summary(finalModel)
anova(finalModel,model)
finalModel <- model

##### 7. Model Evaluation #####

### Train Set
loansTrain$pred <- predict(finalModel, type="response")
predAux <- prediction(loansTrain$pred, loansTrain$loan_status, label.ordering = NULL)
auc.tmp <- performance(predAux, "auc")
aucModeloLogitTrain <- as.numeric(auc.tmp@y.values)
aucModeloLogitTrain

CurvaRocModeloLogitTrain <- performance(predAux,"tpr","fpr")
plot(CurvaRocModeloLogitTrain,colorize=TRUE)
abline(a=0,b=1)

## GINI index
GINItrain <- 2*aucModeloLogitTrain-1
GINItrain


### Test Set
loansTest$pred <- predict(finalModel, newdata = loansTest, type="response")
predAux <- prediction(loansTest$pred, loansTest$loan_status, label.ordering = NULL)
auc.tmp <- performance(predAux, "auc")
aucModeloLogitTest <- as.numeric(auc.tmp@y.values)
aucModeloLogitTest

CurvaRocModeloLogitTest <- performance(predAux,"tpr","fpr")
plot(CurvaRocModeloLogitTest,colorize=TRUE)
abline(a=0,b=1)

## GINI index
GINItest <- 2*aucModeloLogitTest-1
GINItest


### Model improve
mean(as.numeric(loansTest$loan_status))
aggregate(loansTest$pred~loansTest$loan_status,FUN=mean)
