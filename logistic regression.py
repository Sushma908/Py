#!/usr/bin/env python
# coding: utf-8

# ## feb 29th ML algorithm

# ## Logistic regression

# In[3]:


import pandas as pd 
data=pd.read_csv('breast_cancer.csv')
print(data.shape)
data.head()


# In[7]:


X=data.iloc[:,:-1]
y=data.outcome


# In[8]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[10]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)


# In[12]:


model.fit(X_train,y_train)
y_predict=model.predict(X_test)


# In[13]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)


# In[17]:


pd.crosstab(y_test,y_predict)


# In[16]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict)) #59 cancer and 112 non cancer


# In[18]:


pd.DataFrame(model.predict_proba(X_test))


# In[ ]:




