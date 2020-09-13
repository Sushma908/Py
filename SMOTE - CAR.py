#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('car_evaluation.csv')
data.head()


# In[3]:


from collections import Counter
Counter(data.outcome)


# In[4]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
data.buying = enc.fit_transform(data.buying)
data.maint = enc.fit_transform(data.maint)
data.lug_boot = enc.fit_transform(data.lug_boot)
data.safety = enc.fit_transform(data.safety)
data.head()


# In[5]:


from sklearn.preprocessing import scale
X = data.iloc[:,:-1]
y = data.outcome


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=10,test_size=0.3)


# In[7]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)


# In[8]:


model.fit(X_train, y_train)


# In[9]:


y_predict = model.predict(X_test)


# In[10]:


Counter(y_test)


# In[12]:


from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test,y_predict))
pd.crosstab(y_test,y_predict)


# In[13]:


from imblearn.over_sampling import SMOTE


# In[19]:


smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_sample(X_train.astype('float'),y_train)
print("before smote:", Counter(y_train))
print("After smote:", Counter(y_train_smote))


# In[21]:


model.fit(X_train_smote, y_train_smote)
y_predict = model.predict(X_test)
print(accuracy_score(y_test,y_predict))
pd.crosstab(y_test,y_predict)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




