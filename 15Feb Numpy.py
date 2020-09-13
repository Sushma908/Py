#!/usr/bin/env python
# coding: utf-8

# # Numpy - Numerical Python
# Array - Homogeneous

# In[1]:


import numpy as np


# In[ ]:


l=[range(10)]


# In[2]:


a = [3,5,9]
print(a)
type(a)


# In[3]:


a = np.array([3.5,5,9])
print(a)
type(a) # N-Dimensional Array


# In[4]:


a = np.array([3,5,9])
print(a)
print("Dimension :",a.ndim)
print("Dtype :",a.dtype)
print("Size :",a.size)
print("Shape :",a.shape) # elements in each Dimension


# In[5]:


# 2-D array is collection of 1-D arrays
a = np.array([ [3,5,9,4] , [4,6,4,7], [4,5,7,3] ] )
print(a)
print("Dimension :",a.ndim)
print("Dtype :",a.dtype)
print("Size :",a.size)
print("Shape :",a.shape) # elements in each Dimension


# In[6]:


# 3-D array is collection of 2-D arrays
a = np.array([
             [[3,5,9,4],[4,6,4,7],[4,5,7,3]],
             [[8,3,2,1],[6,6,8,7],[3,8,3,6]]
             ])
print(a)
print("Dimension :",a.ndim)
print("Dtype :",a.dtype)
print("Size :",a.size)
print("Shape :",a.shape) # elements in each Dimension


# In[7]:


a


# In[8]:


a[1][0][2] # [1] - 3D index, [0] - 2-D index..


# In[9]:


a[1,0,2]


# # np.arange( )
# Generates a sequencial array

# In[10]:


np.arange(10)


# In[11]:


np.arange(10,20)


# In[12]:


np.arange(10,20,2)


# In[13]:


print(np.arange(10,20,1.5))
print(np.arange(-10,20,2))
print(np.arange(30,20,-2))


# # reshape( )

# In[14]:


a = np.arange(24)
a


# In[15]:


a.reshape(4,3,2)


# In[16]:


a.ndim


# In[17]:


a.reshape(4,3,2)


# ## arange(24) make shape of 6-D

# In[18]:


a = np.arange(24)
a= a.reshape(1,1,1,1,1,1,1,2,3,4)
print(a)
print('Dimension :',a.ndim)
print('Shape :',a.shape)


# # ravel( )

# In[19]:


a


# In[20]:


a.ravel()


# # np.random.random()

# In[21]:


np.random.random()

min + (max-min)* random()
35+ (135)*random()
# In[22]:


int(35 + (135)*np.random.random())


# In[23]:


np.random.randint(35,100,10)


# In[24]:


marks = np.random.randint(35,100,100).reshape(20,5)


# In[25]:


np.sqrt(marks)


# In[26]:


np.sqrt(np.random.randint(35,100,100000000).reshape(10000000,10))


# In[27]:


np.zeros(10).reshape(2,5)


# In[28]:


np.floor(4.7) # floor returns largest integer that is less than x value. ceil returns integer that is greater than x


# In[29]:


np.ceil(4.3)


# In[30]:


np.round(4.50)#round off


# In[31]:


a = np.random.randint(1,10,9).reshape(3,3)
b = np.random.randint(1,10,9).reshape(3,3)
print(a)
print(b)


# In[32]:


np.hstack( (a,b))#horizontal reshaping


# In[33]:


np.vstack( (a,b))#vertical reshaping


# In[34]:


a = np.random.randint(1,12,20).reshape(5,4) # between 1 to 12, 20 elements
a


# In[35]:


np.hsplit(a,2)[1]


# In[36]:


a = np.random.randint(1,10,4).reshape(2,2)
b = np.random.randint(1,10,4).reshape(2,2)
a,b


# In[37]:


np.matmul(a,b)#multiplication of two matrix


# ### Data Treatement,Preparation,Wrangling,

# ## missing values

# In[40]:


import pandas as pd


# In[41]:


data = pd.read_csv('mtcars_missing.csv')
data.head()


# In[42]:


# NAN-not a number,NAt -not a time,none


# In[43]:


np.pi


# In[91]:


np.tan(0.5)


# In[92]:


np.exp(3)


# In[44]:


import numpy as np
np.nan #nan is a special object thatrepresents missing value


# In[45]:


5==5


# In[46]:


np.nan == np.nan


# In[96]:


data.head()


# In[97]:


data.isnull().sum() #gives a nan value in a dataset in column wise


# In[98]:


data.info()#it tells how many rows and columns and how many are int,float you have and how many non null values you have


# In[99]:


data.disp.sort_values(ascending=True) # $


# In[100]:


data.wt.sort_values(ascending=True) # ?


# In[101]:


data.qsec.sort_values(ascending=True) # -


# In[102]:


data.replace(['$','?','*','-'],np.nan,inplace=True)


# In[103]:


data.head()


# In[104]:


data.isnull().sum()


# In[105]:


data.describe()


# In[108]:


data.disp=data.disp.astype('float')
data.wt=data.wt.astype('float')
data.carb=data.carb.astype('float')


# In[109]:


data.head()


# In[110]:


data.isnull().sum().sum()#to find how may missing values are there


# In[111]:


data.describe()


# In[122]:


data.loc[data.mpg>100,'mpg'] = np.nan


# In[60]:


data.describe()


# # NaN treatement

# In[112]:


data.head()


# In[113]:


data.fillna(100)


# In[114]:


data.fillna(method = 'ffill') #forward fill(NaN filled with above records-copying from above record)


# In[115]:


data.fillna(method = 'bfill') #backward fill(NaN filled with above records-copying from back record)ffill and bfill used for categorical vairables


# In[116]:


data.fillna(data.mean()) #mean imputation(automatically calculates the mean)used for continuous variables
#mean imputation is popular for continous variables(because it reduces error)

 predivting imputation is better than mean imputation is most cases
1.when an important column has significant missing values.
2.the x variables used in predictive imputation is not recomended for final modeling
# In[124]:


data.dropna().shape #drops all the records which is having missing values(out of 28records 6 are having missing values)


# In[125]:


(data.shape[0]-data.dropna().shape[0])/data.shape[0] #12% missing values in records


# # Duplicates

# In[126]:


data=pd.read_csv('mtcars_duplicates.csv')
data.head()


# In[136]:


data[10:11]


# In[137]:


data.drop_duplicates(['wt','qsec']).shape


# In[141]:


data.drop_duplicates(['wt','qsec'],inplace=True)
data


# # sorting

# In[145]:


data.sort_values('disp') #sorting according to mileage


# In[143]:


data.sort_values('mpg',ascending=False) #to sort in descending order


# In[144]:


data.sort_values(['mpg','qsec','hp'],ascending=[True,False,True])#mpg ascending,qsec in descending and hp in ascending


# In[ ]:





# In[ ]:




