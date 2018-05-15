
# coding: utf-8

# Processing Variables For Modeling
# ==================

# In[181]:


# remove warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt; plt.style.use("ggplot")

from collections import Counter


# In[2]:


loans = pd.read_csv("../data/clean/loans.csv", sep = "^").sample(200000)


# ### 01 - Target: Loan Status

# `loan_status` is the current status of the loan. This is the variable we want to predict in our machine learning model. For this variable, we are going to considerar three labels:
# - 0: loans that have already been paid.
# - 1: default or charged off loans.
# - 2: current loans (rest of the cases), where we don't know if they are going to be paid or not.
# 
# We will use labels 0 and 1 for training and testing our model. Label 2 is going to use just for predicting.

# In[3]:


loans['loan_status'].value_counts()


# In[4]:


def process_loan_status(loan_status):
    
    loan_status_dict = {
    "Current": 2,
    "Fully Paid": 0,
    "Charged Off": 1,
    "Late (31-120 days)": 2,
    "In Grace Period": 2,
    "Late (16-30 days)": 2,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status:Charged Off": 1,
    "Default": 1
    }
    
    return loan_status_dict[loan_status]


# In[5]:


loans['loan_status'] = loans['loan_status'].map(process_loan_status).                            astype('category')


# In[6]:


loans['loan_status'].describe()


# ### 02 - Categorical Variables

# In[7]:


categorical_variables = ['term', 'grade', 'emp_title', 'emp_length', 'title', 
                         'home_ownership', 'zip_code', 'addr_state', 'application_type']


# In[8]:


for variable in categorical_variables:
    number_of_categories = loans[variable].unique().size
    print("{}: {} categories".format(variable, number_of_categories))


# Too many categories for `emp_title`, `title` and `zip_code`. Let's take a look at these three variables:

# __emp_title__:

# In[9]:


loans['emp_title'].value_counts().head(10)


# In[10]:


loans.loc[~((loans['emp_title'] == 'Teacher') |
            (loans['emp_title'] == 'Manager') | 
            (loans['emp_title'] == 'Owner')),'emp_title'] = 'Other'


# In[11]:


loans['emp_title'].value_counts()


# __title__:

# In[12]:


loans['title'].value_counts().head(20)


# In[13]:


loans = loans.drop('title', axis=1)


# __zip_code__:

# In[14]:


loans['zip_code'].describe()


# In[15]:


loans = loans.drop('zip_code', axis=1)


# __Transform to categorical__:

# In[16]:


categorical_variables = ['term', 'grade', 'emp_title', 'emp_length', 
                         'home_ownership', 'addr_state', 'application_type']


# In[17]:


for variable in categorical_variables:
    loans[variable] = loans[variable].astype("category")


# In[18]:


loans[categorical_variables].describe()


# #### 3.3 - Dates

# We have just one date variable, `issue_d`. We are only interested in the year of the loan and we will consider it as categorical variable:

# In[19]:


loans['issue_d'] = loans['issue_d'].map(lambda x: x[4:])


# In[20]:


loans['issue_d'] = loans['issue_d'].astype('category')


# In[21]:


loans['issue_d'].describe()


# In[45]:


categorical_variables = ['issue_d', 'term', 'grade', 'emp_title', 'emp_length', 
                         'home_ownership', 'addr_state', 'application_type']


# #### 3.4 - Numeric Variables

# In[22]:


numerical_variables = ["funded_amnt_inv", "installment", "int_rate", "annual_inc", "dti",
                       "last_pymnt_amnt", "total_pymnt_inv", "total_rec_late_fee", "total_acc"]


# The only variable we are going to process in this part is the interest rate on the loan (`int_rate`). We have to take the number without the percentage symbol and then transform to float:

# __int_rate__:

# In[23]:


loans['int_rate'] = loans['int_rate'].map(lambda x: float(x[:-1]))


# In[24]:


loans.dtypes


# __outliers detection__:

# In[25]:


loans[numerical_variables].describe()


# In[26]:


def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers


# In[27]:


# detect outliers from numerical features 
outliers_to_drop = detect_outliers(loans,2,numerical_variables)

print("There are {} outliers from numerical features".format(len(outliers_to_drop)))


# In[32]:


loans = loans.drop(outliers_to_drop, axis=0)


# In[33]:


loans.shape


# In[34]:


loans[numerical_variables].describe()


# #### 3.5 - Dealing with NA's

# In[35]:


loans.isnull().sum()


# In[36]:


loans = loans.fillna(method = 'ffill')


# In[37]:


loans.isnull().sum()


# ### 04 -Model

# In[175]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_fscore_support


# In[55]:


data_for_modeling = loans[(loans['loan_status'] == 0) |
                          (loans['loan_status'] == 1)]


# In[56]:


data_for_modeling.shape


# In[57]:


data_for_modeling.dtypes


# In[58]:


data_for_modeling.head()


# In[59]:


categorical_variables


# In[60]:


categorical_for_modeling = ['term', 'grade', 'emp_title', 'emp_length', 'home_ownership', 
                            'application_type']


# In[61]:


data_for_modeling = pd.get_dummies(data_for_modeling, columns = categorical_for_modeling)


# In[62]:


data_for_modeling = data_for_modeling.drop(["issue_d", "addr_state"], axis = 1)


# In[63]:


data_for_modeling.head()


# In[47]:


numerical_variables


# In[212]:


data_for_modeling = data_for_modeling.drop(['last_pymnt_amnt', 'total_pymnt_inv'], axis = 1)


# In[213]:


data_for_modeling.head()


# __Train / Test split:__

# In[214]:


X = data_for_modeling.loc[:, data_for_modeling.columns!='loan_status']


# In[215]:


y = data_for_modeling['loan_status']


# In[216]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# __Prior__:

# In[217]:


y.value_counts()[0] / len(y)


# __Random Forest__:

# In[218]:


random_forest = RandomForestClassifier()


# In[219]:


random_forest.fit(X_train, y_train)


# Feature importance:

# In[220]:


features = pd.DataFrame({'feature':X_train.columns,
                         'importance':random_forest.feature_importances_})

features = features.set_index('feature').sort_values('importance', ascending=True)

features.plot(kind='barh',figsize=(15,15)).set_title('Feature importance')


# __Model Evaluation__

# ROC Curve:

# In[235]:


y_scores = random_forest.predict_proba(X_test)

y_scores = pd.DataFrame(y_scores).loc[:,1]


# In[236]:


fpr, tpr, _ = roc_curve(y_test, y_scores)


# In[237]:


plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")


# AUC:

# In[225]:


np.array(y_scores)


# In[226]:


np.array(y_test)


# In[227]:


roc_auc_score(y_test, y_scores)


# Confusion Matrix

# In[228]:


y_predictions = random_forest.predict(X_test)


# In[229]:


confusion_matrix(y_test, y_predictions)


# In[238]:


(14689+704)/(14689+704+3371+737)


# Recall:

# In[230]:


precision_recall_fscore_support(y_test, y_predictions) [0]


# Precision:

# In[231]:


precision_recall_fscore_support(y_test, y_predictions) [1]

