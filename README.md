# Loan Default Prediction System

Lending Club is a peer to peer lending company based in the United States, in which investors provide funds for potential borrowers and investors earn a profit depending on the risk they take (the borrowers credit score). Lending Club provides the "bridge" between investors and borrowers.
For more information about the company, please check out the official [website](www.lendingclub.com)

__Data Acquistion__

Lending Club provides several csv files that contain complete loan data for all loans issued from 2007 to last 2017 quarter, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. Some additional features such as credit scores, number of finance inquiries, zip codes, states, or collections have been included among others.
Lending club provides a Data Dictionary that includes definitions for all the data attributes included in the Historical data file and the In Funding data file.

__Objective__

The scope of the project is to develop a scoring system consisting of several machine learning models to predict whether a loan will be paid or not.

_Note:_ English is not my native language, so sorry if you see any mistake and do not hesitate to let me know

## Repository Structure

The organization of the repository is divided into the following folders:
- __doc__: this folder is made up of several documentation Jupyter Notebooks of the main stages of the project, except the souce code main program.
- __src__: source code scripts. Notebooks are used as first exploration to the data and to the model hyperparametrizations and first results. But once the code guidelines was more or less clear, it is implemented in the main program.
- __data__: Lending Club data dictionary and features we have decided to keep for feeding the models. The dataset is not available in this folder due to the data is too big. 
- __output__: metrics of the model and model objects.
- __dashboard__ with the main results of the models.

## About the methodology

__Programming lenguages__:

- __Linux shell__: Shell was used mainly to manage files and run scripts.
- __R__: It was used as a first quick approach to the project and to perform some hypothesis tests.
- __Python__ for the Exploratory Data Analysis and the modeling phase. __Jupyter Notebooks__ were used as first exploration to the data and to the model hyperparametrizations over the sample data set and __Spyder__ IDE for source code implementation. Python was also used to deploy prediction system as API. __Anaconda__ distribution was used for creating workflows and keeping the dependencies separated out.
- __Tableau__ for visualization main results of the models.

__Main Libraries__:

- `dplyr`: this package provides simple “verbs”, functions that correspond to the most common data manipulation tasks, to help you translate your thoughts into codeeasy-to-use data structures and perfect for quick data analysis in R.
- `pandas`: a high-performance library, easy-to-use data structures and data analysis tools for the Python programming language.
- `seaborn`: for statistical data __visualizations__ based on `matplotlib`.
- `sklearn`: machine learning library that provides efficient tools for data mining and data analysis built on `NumPy` and `SciPy`. __Preprocessing__ module was used for feature extraction and normalization and __Model selection__ module for comparing, validating and choosing parameters and models. Logistic Regression and Random Forest algorithms from __Classification__ module as well, while `xgboost` library was used for __XG Boost__ algorithm.
- `TensorFlow`: A Deep Learning library that has been used for __Neural Network Autoencoder__ for feature extraction.
- `flask` and `gunicorn`: to deploy Machine Learning models in Production as API.

## How to run this analysis
__1. Creating a virtual environment with Anaconda__ (using _environment.yml_):
```
conda env create -f environment.yml -n environmentName
```

__2. Cleaning and processing data__ (setting local path files):
```
python src/cleaning_data.py
```
__3. Training new data__:
```
python src/main.py
```
__4. Querying via API__:

- Run the API locally:
```
gunicorn --bind 0.0.0.0:8000 src.server:app
```
- API request with Python (where _new_data_ is the data for prediction in json format):
```
requests.post("http://0.0.0.0:8000/predict",
              data = json.dumps(new_data),
              headers = {'Content-Type': 'application/json',
                         'Accept': 'application/json'}).json()
```
- This could be an example of input json to call the API:
```
[
  {
    "addr_state": "NM",
    "annual_inc": 50000.0,
    "application_type": "Individual",
    "avg_cur_bal": 27505.0,
    "bc_open_to_buy": 1312.0,
    "bc_util": 62.5,
    "delinq_2yrs": 0.0,
    "delinq_amnt": 0.0,
    "disbursement_method": "Cash",
    "dti": 12.34,
    "emp_length": "10+ years",
    "emp_title": "him specialist ii",
    "fico_range_high": 689.0,
    "fico_range_low": 685.0,
    "funded_amnt": 15000.0,
    "funded_amnt_inv": 15000.0,
    "grade": "C",
    "home_ownership": "MORTGAGE",
    "id": 166105,
    "initial_list_status": "w",
    "installment": 517.34,
    "int_rate": 14.64,
    "loan_amnt": 15000.0,
    "mort_acc": 3.0,
    "num_accts_ever_120_pd": 0.0,
    "num_bc_sats": 1.0,
    "num_bc_tl": 1.0,
    "num_il_tl": 13.0,
    "num_op_rev_tl": 4.0,
    "num_rev_accts": 4.0,
    "num_rev_tl_bal_gt_0": 2.0,
    "num_sats": 9.0,
    "open_acc": 9.0,
    "pct_tl_nvr_dlq": 90.0,
    "percent_bc_gt_75": 0.0,
    "pub_rec": 0.0,
    "pub_rec_bankruptcies": 0.0,
    "purpose": "debt_consolidation",
    "revol_bal": 2273.0,
    "revol_util": 34.4,
    "sub_grade": "C3",
    "tax_liens": 0.0,
    "term": " 36 months",
    "tot_coll_amt": 101.0,
    "tot_hi_cred_lim": 260757.0,
    "total_bal_ex_mort": 35822.0,
    "total_bc_limit": 3500.0,
    "total_il_high_credit_limit": 32556.0,
    "verification_status": "Source Verified",
    "zip_code": "870xx"
  }
]
```
The meaning of each field can be found in the __`data/features_dict.xlsx`__ data dictionary file or in the __`doc/01-getting_and_cleaning_data.ipynb`__ notebook.

And this is the response of the API in the example above:
```
[
  {
    "id": 166105,
    "logit": 0.2610819251,
    "rf": 0.1888843702,
    "xg": 0.3054217696
  }
]
```
Where __`id`__ is the ID of the loan and __`logit`__, __`rf`__ and __`xg`__ are the scores of the logistic regression, random forest and xg boost models respectively.

## About the author:

__Juan Antonio Morales__ Data Scientist
* https://es.linkedin.com/in/juan-antonio-morales-jiménez-4052593b
* juanan4290@gmail.com

_“All models are wrong but some are useful”_ – George Box
