#!/usr/bin/env python
# coding: utf-8

# <h1>Task 1<h1>

# <h1>Iris Flower Classification

# In[1]:


#importing all required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[44]:


#load the dataset
dataset = pd.read_csv(r"C:\Users\user\Desktop\iris.csv")


# In[45]:


dataset.head() #first five records


# In[46]:


dataset.tail() #last five records


# <h3>VISUALIZATION OF OUR DATASET

# In[47]:


dataset.describe()


# In[48]:


#describe the shape of dataset
dataset.shape


# In[49]:


#giving the total number of rows and columns
print("Number of rows",dataset.shape[0])
print("Number of colums",dataset.shape[1])


# In[62]:


dataset.info() #gives the information related to the dataset 


# In[63]:


#find correlation
dataset.corr()


# In[64]:


sns.pairplot(dataset)#clear the relationship between pairs of features of iris flower


# <h3>separate features and target

# In[65]:


#separate features and target gives the last value 
data=dataset.values
X=data[:,0:4]
Y=data[:,4]
print(X)


# <h3> Label Encoder

# In[66]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[67]:


dataset['Species']=le.fit_transform(dataset['Species'])
dataset.head()


# <h2> Model traning</h2>

# In[68]:


from sklearn.model_selection import train_test_split
x=dataset.drop(columns=['Species'])
y=dataset['Species']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


# <h4>decision Tree Classifier

# In[69]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[70]:


#model traning
model.fit(x_train,y_train)


# In[71]:


predicatList=model.predict(x_test)


# In[72]:


#Print accuracy
print("Accuracy:",model.score(x_test,y_test)*100)


# In[ ]:




