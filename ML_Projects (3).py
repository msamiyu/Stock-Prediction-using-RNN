#!/usr/bin/env python
# coding: utf-8

# #Naive Bayes Classifier IRIS DATASET

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df= pd.read_csv("Iris.csv")


# In[3]:


df.head()


# In[4]:


df.nunique()


# In[5]:


df['Species'].value_counts()


# In[6]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[7]:


df.corr()


# In[8]:


sns.heatmap(df.corr())


# In[9]:


sns.countplot('Species',data = df)


# In[10]:


sns.barplot(x='Species',y='SepalLengthCm',data=df)


# In[11]:


sns.boxplot('Species','SepalLengthCm',data=df, palette='rainbow')


# In[12]:


sns.pairplot(df)


# In[ ]:





# In[13]:


# Train_Test Split


# In[14]:


X = df.iloc[:,:4].values
y = df['Species'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[15]:


#Training the Train_set data
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[16]:


#Predicting the Test set
y_pred = classifier.predict(X_test) 
y_pred


# In[17]:


#Model Performance/ Accuracy


# In[18]:


from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, y_pred))
conf_matrix


# In[19]:


print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(classifier.score(X_train, y_train)))
print('Accuracy of GaussianNB classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))


# In[20]:


#PROJECT 2


# In[ ]:




