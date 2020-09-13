#!/usr/bin/env python
# coding: utf-8

# In[28]:


##  k means 
import pandas as pd
data=pd.read_csv('iris.csv')
print(data.shape)
data.head()
#50 setosa(0),50 versicolor(1),50 verginica (2)


# In[29]:


X=data.iloc[:,:-1]
X.head()


# In[15]:


from sklearn.cluster import KMeans
model=KMeans(n_clusters=3,random_state=10)


# In[16]:


model.fit(X)


# In[17]:


model.fit(X)
model.labels_


# In[18]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


import numpy as np
colorscheme =np.array(['red','green','blue'])


# In[22]:


colorscheme[2]


# In[27]:


plt.scatter(X.sepal_length,X.petal_width,color=colorscheme[data.target]);


# In[32]:


model.cluster_centers_# 4 dimensions so 4 clusters or centroids


# In[33]:


data.head()


# In[34]:


data[data.petal_length<=4.0].target.value_counts()


# In[ ]:




