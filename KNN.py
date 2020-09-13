#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv('car_evaluation.csv')
print(data.shape)
data.head()


# In[2]:


data.outcome.value_counts()


# In[3]:


X=data.iloc[:,:-1]
y=data.outcome
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
X.buying=enc.fit_transform(X.buying)
X.maint=enc.fit_transform(X.maint)
X.lug_boot=enc.fit_transform(X.lug_boot)
X.safety=enc.fit_transform(X.safety)


# In[4]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)


# In[17]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=7)
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)


# In[7]:


pd.crosstab(y_test,y_predict)


# In[18]:


import pandas as pd
data=pd.read_csv('wine_quality_class_cat.csv')
print(data.shape)
data.head()


# In[21]:


data.Quality.value_counts()


# In[25]:


X=data.iloc[:,:-1]
y=data.Quality
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
X.density=enc.fit_transform(X.density)
X.pH=enc.fit_transform(X.pH)
X.sulphates=enc.fit_transform(X.sulphates)
X.alcohol=enc.fit_transform(X.alcohol)


# In[26]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)


# In[38]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1)
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)


# In[45]:


import pandas as pd
data=pd.read_csv('wine_quality_class_cat.csv')
print(data.shape)
data.head()


# In[48]:


X = data.iloc[:,:-1]
y = data.Quality
X.head()


# In[51]:


#Defining and fitting
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3,random_state=1)
model.fit(X,y)


# In[52]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)


# In[66]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(max_depth=1,random_state=1,criterion='entropy')
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)


# In[ ]:





# In[ ]:




