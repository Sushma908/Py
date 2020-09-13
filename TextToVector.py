#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# ### Bag of Words

# In[2]:


messages = ['call you tonight', 'Call me a cab', 'please call me.. please']

Process of converting free text to structured data is called as
Text Vectorization.
# In[3]:


# instantiate CountVectorizer (vectorizer)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(messages)
vect.get_feature_names() # unique words from text


# <h3>Transform message </h3> (Bag of words)

# In[4]:


messages_transformed = vect.transform(messages)
print(messages)
print(vect.get_feature_names())
messages_transformed.toarray()


# In[5]:


data = pd.DataFrame(messages_transformed.toarray())
data.columns = vect.get_feature_names()
print(messages)
data.head()


# In[6]:


data.loc[0,'outcome'] ='info'
data.loc[1,'outcome'] ='order'
data.loc[2,'outcome'] = "request"


# In[7]:


data.head()


# In[ ]:





# In[ ]:





# ### Tfidf - Term Frequency inverse Document Frequency

# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfvect = TfidfVectorizer()
trans = tfvect.fit_transform(messages)
pd.DataFrame(trans.toarray(),columns=tfvect.get_feature_names())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




