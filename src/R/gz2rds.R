gzipFile <- gzfile('/media/juanan/DATA/loan_data_analysis/data/clean/loans.csv.gz','rt')   
data <- read.csv(gzipFile, header = T)

saveRDS(data, '/media/juanan/DATA/loan_data_analysis/data/clean/loans.rds')
