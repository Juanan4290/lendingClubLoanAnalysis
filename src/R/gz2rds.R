gzipFile <- gzfile('/media/juanan/DATA/loan_data_analysis/data/raw/accepted_2007_to_2017Q3.csv.gz','rt')   
data <- read.csv(gzipFile, header = T)

saveRDS(data, '/media/juanan/DATA/loan_data_analysis/data/raw/raw_loans_2007_to_2017Q3.rds')
