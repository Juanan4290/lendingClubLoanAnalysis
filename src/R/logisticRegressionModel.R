##### 1. Libraries #####

library(caTools)
library(dplyr)
library(stringr)
library(ROCR)


##### 2. Read Data #####

loans <- read.csv("~/Documentos/ja-github/loan-data-analysis/data/clean/loans.csv", sep="^")

head(loans)
str(loans)


##### 3. Cleaning Data #####

set.seed(4290)
loans <- loans %>% 
  filter((loan_status == "Fully Paid")|(loan_status == "Charged Off")) %>% 
  sample_n(100000)

loans$loan_status <- as.character(loans$loan_status)

loans$loan_status[loans$loan_status == "Fully Paid"] <- 0
loans$loan_status[loans$loan_status == "Charged Off"] <- 1

loans$loan_status <- as.numeric(loans$loan_status)

loans$int_rate <- sapply(loans$int_rate, function(i) str_sub(i, 1, -2) %>% 
                           as.numeric)

columnsToRemove <- c("issue_d", "emp_title", "title", "zip_code", "addr_state")
loans <- loans[,!(colnames(loans) %in% columnsToRemove)]
loans <- loans[complete.cases(loans),]

##### 4. Dataset summary #####

head(loans)
str(loans)
summary(loans)


##### 5. Train / Test split #####

set.seed(4290)
sampleSplit = sample.split(loans$loan_status, SplitRatio = .75)
loansTrain = subset(loans, sampleSplit == TRUE)
loansTest = subset(loans, sampleSplit == FALSE)


##### 6. Model and feature selection #####

model=glm(loan_status~., data=loansTrain,family=binomial(link="logit"))
summary(model)

finalModel=step(model,direction="both",trace=1)
summary(finalModel)
anova(finalModel,model)


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