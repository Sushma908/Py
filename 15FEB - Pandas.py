#!/usr/bin/env python
# coding: utf-8

# # Pandas  - Panel Data
# Series, DataFrame

# In[1]:


import pandas as pd


# In[2]:


myexp = [40,60,65,200,250,45,55]
myexp


# In[3]:


myexp_series = pd.Series([40,60,65,200,250,45,55])
print(myexp_series)
type(myexp_series)


# In[4]:


myexp_series.index = ['wed','thu','fri','sat','sun','mon','tue']
myexp_series


# In[5]:


myexp_series['sun']


# In[6]:


myexp_series[0:4]


# In[7]:


# retrieving data by Label Slicing
myexp_series['wed':'sun'] # label slicing includes end pos aswell


# In[8]:


# retrieve data by list
myexp_series[['sun','wed','thu']]


# In[9]:


myexp_series[[4,0,1]]


# In[10]:


# Retrieve by condition, SCALAR filtering
myexp_series[ myexp_series>100 ]


# In[18]:


myexp_series[ (myexp_series>50)  & (myexp_series<100) ]


# In[19]:


myexp_series[ (myexp_series<50)  | (myexp_series>100) ]


# # DataFrame

# In[25]:


data = pd.read_csv('mtcars.csv')
type(data)


# In[26]:


data.head()


# In[28]:


pwd


# In[29]:


data.head(3)


# In[32]:


data.loc[2,'mpg'] # label location


# In[31]:


data.iloc[2,1] # iloc index location


# In[34]:


data.loc[0:2,'car_model':'wt']


# In[35]:


data.loc[0:2,['car_model','wt','mpg','hp']]


# In[36]:


data.iloc[0:2,[0,3,5]]


# In[39]:


# Scalar filtering in DataFrame
data.loc[data.mpg>30, : ]


# In[ ]:





# In[ ]:





# In[ ]:





# ## Find which has mpg > 30, hp> 110

# In[40]:


data.loc[(data.mpg>30) & (data.hp>110), : ]


# In[42]:


data[(data.mpg>30) & (data.hp>110)] 
# we use loc/iloc only when we operate  on columns


# In[ ]:





# In[ ]:




