library(dplyr)

# Read Data
loans = readRDS("/media/juanan/DATA/loan_data_analysis/data/clean/loans.rds")

# Selecting fico columns and dates (loan issued and last payment)
ficoStudy <- loans %>% 
  select(issue_d, last_pymnt_d, 
         last_fico_range_low, last_fico_range_high, fico_range_low, fico_range_high, 
         loan_status) %>% 
  mutate(loanInProgress = as.numeric(loan_status == "Current"))

# fico mean for finished and current loans
ficoStudy %>% 
  group_by(loanInProgress) %>% 
  summarise(meanLastFico = mean(last_fico_range_high, na.rm = TRUE), 
            meanFico = mean(fico_range_high, na.rm = TRUE),
            counter = n())

# ratio current/finished loans by date issued
ficoStudy %>% 
  group_by(issue_d) %>% 
  summarise(ratioCurrentFinished = mean(loanInProgress)) %>% 
  View

### t-tests

## Current Loans:
# All current loans:
currentLoans <- ficoStudy %>% 
  filter(loanInProgress == 1)

t.test(currentLoans$last_fico_range_low, currentLoans$fico_range_low,
       paired=F,var.equal=F)

t.test(currentLoans$last_fico_range_high, currentLoans$fico_range_high,
       paired=F,var.equal=F)

# Current loans at september-2016
currentAtSept2016 <- ficoStudy %>% 
  filter(issue_d == "Sep-2016", loanInProgress == 1)

t.test(currentAtSept2016$last_fico_range_low, currentAtSept2016$fico_range_low,
       paired=F,var.equal=F)

t.test(currentAtSept2016$last_fico_range_high, currentAtSept2016$fico_range_high,
       paired=F,var.equal=F)

## Finished Loans:
# All paid loans:
paidLoans <- ficoStudy %>% 
  filter(loan_status == "Fully Paid")

t.test(paidLoans$last_fico_range_low, paidLoans$fico_range_low,
       paired=F,var.equal=F)

t.test(paidLoans$last_fico_range_high, paidLoans$fico_range_high,
       paired=F,var.equal=F)

# All charged off loans:
chargedLoans <- ficoStudy %>% 
  filter(loan_status == "Charged Off")

t.test(chargedLoans$last_fico_range_low, chargedLoans$fico_range_low,
       paired=F,var.equal=F)

t.test(chargedLoans$last_fico_range_high, chargedLoans$fico_range_high,
       paired=F,var.equal=F)


# Paid loans at september-2016
paidAtSept2016 <- ficoStudy %>% 
  filter(issue_d == "Sep-2016", loan_status == "Fully Paid")

t.test(paidAtSept2016$last_fico_range_low, paidAtSept2016$fico_range_low,
       paired=F,var.equal=F)

t.test(paidAtSept2016$last_fico_range_high, paidAtSept2016$fico_range_high,
       paired=F,var.equal=F)


# Charged Off loans at september-2016
chargedAtSept2016 <- ficoStudy %>% 
  filter(issue_d == "Sep-2016", loan_status == "Charged Off")

t.test(chargedAtSept2016$last_fico_range_low, chargedAtSept2016$fico_range_low,
       paired=F,var.equal=F)

t.test(chargedAtSept2016$last_fico_range_high, chargedAtSept2016$fico_range_high,
       paired=F,var.equal=F)


