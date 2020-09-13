#!/usr/bin/env python
# coding: utf-8

# #  Naive Bayes Classifiers

# In[1]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


# ## Naive Bayes
# ### Using Naive Bayes to predict spam

# In[2]:


#Use Latin encoding as the Data has non UTF-8 Chars
data = pd.read_csv("spam.csv",encoding='latin-1')
print(data.shape)
data.head()


# In[3]:


data.email[2]


# In[4]:


X1 =  data.email
y = data.type


# ## Vectorization : Transforming TEXT to Vectors

# In[5]:


vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(X1)
feature_names = vectorizer.get_feature_names()


# In[6]:


len(feature_names)


# In[7]:


feature_names[2000:2010]


# In[8]:


X = X.toarray()


# In[9]:


X.shape


# In[10]:


y.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)


# In[ ]:


#Fitting Naive Bayes algo
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
model = BernoulliNB()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import classification_report
print(accuracy_score(y_test,y_predict))
print(classification_report(y_test,y_predict))
pd.crosstab(y_test,y_predict)


# In[ ]:


X_train[0:10]


# ## Checking new email for spam

# In[ ]:


#NewEmail = pd.Series(["Hi team, We have meeting tomorrow"])
#NewEmail = pd.Series(['**FREE MESSAGE**Thanks for using the Auction Subscription Service. 18 . 150p/MSGRCVD 2 Skip an Auction txt OUT. 2 Unsubscribe txt STOP CustomerCare 08718726270'])
NewEmail = pd.Series(['Hi .. This is Ashok Veda. Are you available this Friday for quick me'])
NewEmail


# In[ ]:


NewEmail_transformed = vectorizer.transform(NewEmail)


# In[ ]:


NewEmail_transformed.shape


# In[ ]:


model.predict(NewEmail_transformed)


# In[ ]:


X_test


# In[ ]:


pd.DataFrame(np.exp(model.feature_log_prob_[0]),index = vectorizer.get_feature_names()).sort_values(0,ascending=False)


# In[ ]:





# In[ ]:





# In[ ]:




