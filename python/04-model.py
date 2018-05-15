
# coding: utf-8

# In[19]:


# remove warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score


# In[3]:


data = pd.read_csv("../data/clean/loans_train_test.csv", sep = "^")


# In[4]:


data.head()


# Train / Test split:

# In[7]:


X = data.loc[:, data.columns!='loan_status']


# In[8]:


y = data['loan_status']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# Classifiers:

# In[14]:


logit_cl = LogisticRegression(random_state=42)


# In[13]:


random_forest = RandomForestClassifier(random_state=42)


# In[17]:


#Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)


# In[20]:


cross_val_score(logit_cl, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4)


# Prior:

# In[25]:


y.value_counts()[0] / len(y)


# In[26]:


cross_val_score(logit_cl, X_train, y = y_train, scoring = "roc_auc", cv = kfold, n_jobs=4)


# In[27]:


cross_val_score(random_forest, X_train, y = y_train, scoring = "roc_auc", cv = kfold, n_jobs=4)

