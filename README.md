# Loan Default Prediction System

Lending Club is a peer to peer lending company based in the United States, in which investors provide funds for potential borrowers and investors earn a profit depending on the risk they take (the borrowers credit score). Lending Club provides the "bridge" between investors and borrowers.
For more information about the company, please check out the official [website](www.lendingclub.com)

__Data Acquistion__

Lending Club provides several csv files that contain complete loan data for all loans issued from 2007 to last 2017 quarter, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. Some additional features such as credit scores, number of finance inquiries, zip codes, states, or collections have been included among others.
Lending club provides a Data Dictionary that includes definitions for all the data attributes included in the Historical data file and the In Funding data file.

__Objective__

The scope of the project is to develop a scoring system consisting of several machine learning models to predict whether a loan will be paid or not.

## Repository Deployment

The organization of the repository is divided into the following folders:
- __doc__: this folder is made up of several documentation Jupyter Notebooks of the main stages of the project, except the souce code main program.
- __src__: source code scripts. Notebooks are used as first exploration to the data and to the model hyperparametrizations and first results. But once the code guidelines was more or less clear, it is implemented in the main program.
- __data__: Lending Club data dictionary and features we have decided to keep for feeding the models. The dataset is not available in this folder due to the data is too big. 
- __output__: metrics of the model and model objects.

## About the methodology

__Programming lenguages__:

- __Linux shell__: Shell was used mainly to manage files and run scripts.
- __R__: It was used as a first quick approach to the project and to perform some hypothesis tests.
- __Python__: It was used for the Exploratory Data Analysis and the modelling phase. __Jupyter Notebooks__ were used as first exploration to the data and to the model hyperparametrizations over the sample data set. __Spyder__ IDE was used for souce code implementation.

__Main Libraries__:

- `dplyr`: this package provides simple “verbs”, functions that correspond to the most common data manipulation tasks, to help you translate your thoughts into codeeasy-to-use data structures and perfect for quick data analysis in R.
- `pandas`: a high-performance library, easy-to-use data structures and data analysis tools for the Python programming language.
- `seaborn`: for statistical data __visualizations__ based on `matplotlib`.
- `sklearn`: machine learning library that provides efficient tools for data mining and data analysis built on `NumPy` and `SciPy`. __Preprocessing__ module was used for feature extraction and normalization and __Model selection__ module for comparing, validating and choosing parameters and models. Logistic Regression and Random Forest algorithms from __Classification__ module as well, while `xgboost` library was used for __XG Boost__ algorithm.
- `TensorFlow`: A Deep Learning library that has been used for __Neural Network Autoencoder__ for feature extraction.

## About the author:

__Juan Antonio Morales__ Data Scientist at Idealista/Data
* https://es.linkedin.com/in/juan-antonio-morales-jiménez-4052593b
* juanan4290@gmail.com
