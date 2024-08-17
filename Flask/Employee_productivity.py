#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import MultiColumnLabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import xgboost as xgb
import pickle


# In[5]:


dataset = pd.read_csv("garments_worker_productivity.csv")
dataset.head()


# In[6]:


dataset.shape


# In[7]:


dataset.info()


# In[8]:


dataset['team'] = dataset['team'].astype('int32')
dataset['over_time'] = dataset['team'].astype('int32')
dataset['incentive'] = dataset['incentive'].astype('int32')
dataset['idle_men'] = dataset['idle_men'].astype('int32')
dataset['no_of_style_change'] = dataset['no_of_style_change'].astype('int32')

dataset['targeted_productivity'] = dataset['targeted_productivity'].astype('float32')
dataset['smv'] = dataset['smv'].astype('float32')
dataset['idle_time'] = dataset['idle_time'].astype('float32')
dataset['no_of_workers'] = dataset['targeted_productivity'].astype('float32')
dataset['actual_productivity'] = dataset['actual_productivity'].astype('float32')


# In[9]:


dataset.info()


# In[10]:


dataset.describe()


# In[11]:


# Number of NULL values
dataset.isna().sum()


# In[12]:


# Drop null values
dataset.drop("wip",axis=1, inplace=True)


# In[13]:


# Number of Non-NULL values
dataset.notna().sum()


# In[14]:


# Making the two date formats uniform and converting them into respective months
dataset['date'] = pd.to_datetime(dataset['date'])
dataset['month'] = dataset['date'].dt.month_name()


# In[15]:


# No need of date column
dataset.drop("date", axis=1, inplace=True)


# In[16]:


dataset


# In[17]:


dataset['quarter'].value_counts()


# In[18]:


dataset['department'].value_counts()


# In[19]:


dataset['department'].unique()


# In[20]:


# Cleaning and fixing dataset for the correct Department value
dataset['department'] = dataset['department'].replace('sweing', 'Sewing')
dataset['department'] = dataset['department'].replace('finishing ', 'Finishing')
dataset['department'] = dataset['department'].replace('finishing', 'Finishing')


# In[21]:


dataset['department'].value_counts()


# In[22]:


dataset['day'].value_counts()


# In[23]:


dataset['month'].unique()


# In[24]:


X = dataset.drop(['actual_productivity'],axis=1)
y = dataset['actual_productivity']


# In[25]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.8,random_state=0)


# In[26]:


X_train


# In[27]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer


# ### Linear Regression

# In[28]:


step1 = ColumnTransformer(transformers=[
    ('tnf1', OneHotEncoder(sparse_output=False , handle_unknown='ignore'), ['quarter','department']),
    ('tnf2', OrdinalEncoder(categories = [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday', 'Sunday']]), ['day']),
    ('tnf3', OrdinalEncoder(categories = [['January', 'February', 'March']]), ['month'])
], remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)

print('Mean squared Error ',mean_squared_error(y_test, y_pred))
print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# #### RandomForest

# In[29]:


step1 = ColumnTransformer(transformers=[
    ('tnf1', OneHotEncoder(sparse_output=False ,handle_unknown='ignore'), ['quarter','department']),
    ('tnf2', OrdinalEncoder(categories = [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday', 'Sunday']]), ['day']),
    ('tnf3', OrdinalEncoder(categories = [['January', 'February', 'March']]), ['month'])
], remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)

print('Mean squared Error ',mean_squared_error(y_test, y_pred))
print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### XGBoost Model

# In[30]:


step1 = ColumnTransformer(transformers=[
    ('tnf1', OneHotEncoder(sparse_output=False ,handle_unknown='ignore'), ['quarter','department']),
    ('tnf2', OrdinalEncoder(categories = [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday', 'Sunday']]), ['day']),
    ('tnf3', OrdinalEncoder(categories = [['January', 'February', 'March']]), ['month'])
], remainder='passthrough')

step2 = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)

print('Mean squared Error ',mean_squared_error(y_test, y_pred))
print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Importing Model

# In[33]:


pickle.dump(dataset,open('dataset.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[32]:


X_test.info()


# In[ ]:




