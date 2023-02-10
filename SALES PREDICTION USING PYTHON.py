#!/usr/bin/env python
# coding: utf-8

# <h1>Task 5

# <h2>SALES PREDICTION USING PYTHON

# In[1]:


# importingall required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#read the file
df=pd.read_csv(r"C:\Users\user\Desktop\Advertising.csv")


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


#Cheak row,column,missing value,datatype
df.info()


# In[7]:


#Overviews the statistical in each variable
df.describe()


# In[34]:


df.hist(bins = 30, figsize = (20, 20), color = 'y')


# In[9]:


sns.pairplot(df)


# In[11]:


#find correlation
correlations = df.corr()
sns.heatmap(correlations, annot = True)


# In[12]:


df.head(5)


# In[15]:



X = df[['TV', 'Radio', 'Newspaper']]
X


# In[17]:


#gives the shape
X.shape


# In[19]:


y = df['Sales']
y


# In[20]:


y.shape


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[22]:


X_train.shape


# In[23]:


X_test.shape


# In[24]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept = True)
regressor.fit(X_train, y_train)


# In[25]:


print('Linear Model Coeff (m):', regressor.coef_)
print('Linear Model Coeff(b):', regressor.intercept_)


# In[26]:


y_predict = regressor.predict(X_test)
y_predict


# In[27]:


y_test


# In[28]:


plt.scatter(y_test, y_predict, color = 'r')
plt.ylabel('Model Predictions')
plt.xlabel('True Value (Ground Truth)')
plt.title('Linear Regression Predictions')
plt.show()


# In[29]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)), '.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
MAPE = np.mean( np.abs( (y_test - y_predict) / y_test ) ) * 100


# In[30]:


print('RMSE =', RMSE, '\nMSE =', MSE, '\nMAE =', MAE, '\nMean Absolute Percentage Error =', MAPE, '%')


# In[ ]:




