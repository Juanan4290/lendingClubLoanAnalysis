# Libraries
library(dplyr)
library(stringr)

# Read Data
loans = readRDS("/media/juanan/DATA/loan_data_analysis/data/raw/raw_loans_2007_to_2017Q3.rds")

# Filter only Fully Paid and Charged Off loans and get a sample of 100000 rows
loans <- loans %>% 
  filter(loan_status=="Fully Paid" | loan_status=="Charged Off") %>% 
  sample_n(100000)

# Transform loan_status to numeric
loans$loan_status <- as.character(loans$loan_status)
loans$loan_status[loans$loan_status=="Fully Paid"] <- 0
loans$loan_status[loans$loan_status=="Charged Off"] <- 1
loans$loan_status <- as.numeric(loans$loan_status)

# NA by column
loans[loans == ""] <- NA
naPercentage <- loans %>% 
  is.na %>%
  colMeans %>% 
  as.numeric

hist(naPercentage, breaks = 50)

# Getting only columns with, at least, 75% values (25% of NA's or less). View distribution
loans <- loans[,naPercentage<0.2]

# categorical and numeric features
categoricalFeatures <- loans[,sapply(loans, is.factor)] %>% 
  colnames
numericFeatures <- loans[, sapply(loans, is.numeric)] %>% 
  colnames
summary(loans[categoricalFeatures])

## Categorical Features
# unique values by column in categorical features for filter those that have more than one unique value and less than 15
uniqueCategorical <- sapply(loans[categoricalFeatures], function(i) unique(i) %>% 
                              length)
oneUniqueCategoricalMask <- !sapply(loans[categoricalFeatures], function(i) unique(i) %>% 
                                      length) == 1
tooManyUniqueCategoricalMask <- !sapply(loans[categoricalFeatures], function(i) unique(i) %>% 
                                          length) > 15

categoricalMask <- (oneUniqueCategoricalMask + tooManyUniqueCategoricalMask) == 2

loansCategorical <- loans[categoricalFeatures[categoricalMask]]


## Numerical Features
summary(loans[numericFeatures])

oneUniqueNumericMask <- !sapply(loans[numericFeatures], function(i) unique(i) %>% 
                                  length) == 1

loansNumerical <- loans[numericFeatures[oneUniqueNumericMask]]

## Clean Data
loans <- cbind(loansCategorical, loansNumerical)

# only complete cases
loans <- loans[complete.cases(loans),]

### Feature selection
  columnsOfInterest <- c("funded_amnt_inv", "term","loan_status", "installment",
                         "int_rate", "grade", "emp_length",
                         "annual_inc", "home_ownership", "dti") 
                         #"fico_range_low", 
                         #"fico_range_high",
                         #"last_fico_range_high",
                         #"last_fico_range_low")
  
  loans <- loans[, names(loans) %in% columnsOfInterest]
  
  # Grade instead of fico range
  loans$grade <- as.character(loans$grade)
  loans$grade[loans$grade=="A"] <- 1
  loans$grade[loans$grade=="B"] <- 2
  loans$grade[loans$grade=="C"] <- 3
  loans$grade[loans$grade=="D"] <- 4
  loans$grade[loans$grade=="E"] <- 5
  loans$grade[loans$grade=="F"] <- 6
  loans$grade[loans$grade=="G"] <- 7
  loans$grade <- as.numeric(loans$grade)
  
  loans %>% 
    select(grade, loan_status) %>% 
    group_by(loan_status) %>% 
    summarise(mean(grade))
  
  ##########################################
  columnsToDrop <- c('collection_recovery_fee', 'debt_settlement_flag', 'last_pymnt_amnt', 'last_pymnt_d', 'recoveries', 
                     'total_pymnt', 'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp', 
                     #"fico_range_low", "fico_range_high", 
                     "last_fico_range_high", "last_fico_range_low")
  
  #otherColumnsToDrop <- c('initial_list_status', )
  
  loans <- loans[ , !(names(loans) %in% columnsToDrop)]
  ##########################################

loans %>% 
    select(last_fico_range_high, last_fico_range_low, fico_range_low, fico_range_high, grade) %>% 
    group_by(grade) %>% 
    summarise(last_high = mean(last_fico_range_high),
              last_low = mean(last_fico_range_low),
              high = mean(fico_range_high),
              low = mean(fico_range_high))
  
# Save data
saveRDS(loans, "data/loans_sample_clean.rds")
