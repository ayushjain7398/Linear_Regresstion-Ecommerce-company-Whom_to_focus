
# coding: utf-8

# ## Imports
# ** Import pandas, numpy, matplotlib,and seaborn. Then set %matplotlib inline 
# (You'll import sklearn as you need it.)**

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[9]:


customers=pd.read_csv('Ecommerce Customers')


# In[11]:


customers.describe()


# In[12]:


customers.info()


# In[279]:





# ## Exploratory Data Analysis
# 
# **Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?**

# In[15]:


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')


# In[16]:


sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data = customers)


# In[281]:





# ** with the Time on App column instead. **

# In[17]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)


# ** Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**

# In[18]:


sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)


# In[19]:


sns.pairplot(customers)


# **Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?**

# In[20]:


#Length of membership


# In[21]:


sns.lmplot(x='Yearly Amount Spent',y='Length of Membership',data=customers)


# ## Training and Testing Data
# 
# 

# In[29]:


X=customers[[ 'Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# In[30]:


y=customers['Yearly Amount Spent']


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Training the Model
# 

# In[34]:


from sklearn.linear_model import LinearRegression


# In[35]:


lm=LinearRegression()


# In[36]:


lm.fit(X_train,y_train)


# In[39]:


print('cofficient: \n',lm.coef_)


# In[41]:


prediction=lm.predict(X_test)


# ** Create a scatterplot of the real test values versus the predicted values. **

# In[42]:


plt.scatter(y_test,prediction)


# ## Evaluating the Model
# 

# In[53]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test,prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# ## Residuals
# 
# 
# **Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().**

# In[55]:


sns.distplot((y_test-prediction),bins=50);


# ## Conclusion
# 
# ** Recreate the dataframe below. **

# In[56]:


pd.DataFrame(lm.coef_,X.columns,columns=['coeff'])


# **Q. How can you interpret these coefficients? **

# In[58]:


Interpreting the coefficients:

Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.


# **Do you think the company should focus more on their mobile app or on their website?**

# 
# *Answer here*

# ## Great Job!
# 
# Congrats on your contract work! The company loved the insights! Let's move on.
