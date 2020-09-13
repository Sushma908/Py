#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv('car_evaluation.csv')
data.head()


# In[2]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
data.buying = enc.fit_transform(data.buying)
data.maint = enc.fit_transform(data.maint)
data.lug_boot = enc.fit_transform(data.lug_boot)
data.safety = enc.fit_transform(data.safety)
X = data.iloc[:,:-1]
y = data.outcome


# In[3]:


X.head()


# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)


# In[16]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
parameters = {'max_depth':[14,15,16],
              'random_state': [0,1,2],
              'n_estimators':[55,60,65], 
             }
#grid = GridSearchCV(model,parameters,cv=5)
grid = RandomizedSearchCV(model,parameters,cv=5)
grid.fit(X,y)


# In[17]:


grid.best_score_


# In[15]:


grid.best_params_


# In[11]:


y_predict = grid.predict(X_test)


# In[12]:


model = RandomForestClassifier(random_state=0,
                               n_estimators=58,
                              max_depth=14)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




