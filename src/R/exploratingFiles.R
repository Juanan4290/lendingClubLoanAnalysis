# libraries
library(dplyr)

# read data
loanFiles <- list.files("~/Documentos/ja-github/loan-data-analysis/data/raw/", full.names = TRUE)

# loading a sample of the first file in the directory
loanSample <- read.csv(loanFiles[1], sep = ",", nrows = 10000)

# Dataset sumary
str(loanSample)
summary(loanSample)

# NA count by columns
na_count <- sapply(loanSample, function(y) sum(length(which(is.na(y))))) %>% 
  data.frame

# keeping top 20 variables (with more than 50% values)
loanSample <- loanSample %>% 
  select(colnames(loanSample)[na_count$. < 5000])

columnsOfInterest <- c("funded_amnt_inv", "term", "issue_d","loan_status", "last_pymnt_d", 
                       "last_pymnt_amnt", "next_pymnt_d", "installment", "total_pymnt_inv",
                       "total_rec_late_fee", "int_rate", "sub_grade", "emp_title", "emp_length",
                       "annual_inc", "home_ownership", "title", "zip_code", "addr_state", "dti")

# Reading columns of interest from all datasets
df <- data.frame(matrix(ncol = length(columnsOfInterest), nrow = 0))
colnames(df) <- columnsOfInterest

# Reading all datasets
for (file in loanFiles){
  
  print(paste("Reading File:", file))
  
  tmpDF <- read.csv(loanFiles[1], sep = ",")
  tmpDF <- tmpDF %>% 
    select(columnsOfInterest)
  
  df <- rbind(df,tmpDF)
}

# Saving DataSet
saveRDS(df, "../../data/loan_clean/allLoanTop20.rds")

# Some exploration of the data set
df$loan_status %>% 
  table

df %>% 
  filter(loan_status == "Late (31-120 days)") %>% 
  select(next_pymnt_d) %>% 
  View