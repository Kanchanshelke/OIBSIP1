#!/usr/bin/env python
# coding: utf-8

# <h1>Task 2

# <h2>UNEMPLOYMENT ANALYSIS WITH PYTHON

# In[16]:


#importing all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[17]:


#load the dataset
data=pd.read_csv(r"C:\Users\user\Desktop\Unemployment in India.csv")
data=pd.read_csv(r"C:\Users\user\Desktop\Unemployment_Rate_upto_11_2020.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# <h3>VISUALIZATION OF OUR DATASET

# In[5]:


data.describe()


# In[6]:


#describe the shape of dataset
data.shape


# In[7]:


#giving the total number of rows and columns
print("Number of rows",data.shape[0])
print("Number of colums",data.shape[1])


# In[8]:


data.info() #gives the information related to the dataset 


# In[9]:


sns.pairplot(data)


# In[10]:


data.isnull().sum()


# In[11]:


data.columns=["States","Date","Frequency","Estimated Unemployment Rate","Estimated Employed",
              "Estimated Labour Participation Rate","Region","longitude","latitude"]


# <h3>Heatmap

# In[12]:


#correlation

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr())
plt.show()


# <h2>Data Visulization

# In[13]:


#Data visualisation
data.columns=["States","Date","Frequency","Estimated Unemployment Rate","Estimated Employed",
              "Estimated Labour Participation Rate","Region","longitude","latitude"]
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Employed", hue="Region",data=data)
plt.show()


# In[14]:


plt.figure(figsize=(12, 10))
plt.title("Unemployment In India")
sns.histplot(x="Estimated Unemployment Rate",hue="Region",data=data)
plt.show()


# <h2>State wise unemployment rate

# In[15]:


plt.figure(figsize=(12, 10))
plt.title("Unemployment In India State wise")
sns.histplot(x="Estimated Unemployment Rate",hue="Region",data=data)
plt.show()


# In[ ]:





# In[ ]:




