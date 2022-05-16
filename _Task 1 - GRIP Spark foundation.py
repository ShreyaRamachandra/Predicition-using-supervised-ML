#!/usr/bin/env python
# coding: utf-8

# # The Spark Foundation-Data Science and Analytics Internship
# 
# __TASK 1: Predicition using Supervised ML__
# 
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied.
# 

# __Steps:__
# 1. Importing the dataset
# 2. Visualising the dataset
# 3. Data Preparation
# 4. Training the algorithm
# 5. Visualising the model
# 6. Making predicitons
# 7. Evaluating the model

# **Author: Shreya Ramachandra**

# __1. Importing the dataset__
# 
# In this step we are importing data set with the help of libraries and reading the data
# 

# In[ ]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(25)


# __2. Visualising the Dataset__
# 
# Plotting the distribution of scores on 2-D graph
# 

# In[27]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# __From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.__

# __3. Data Preparation__
# 
# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs). the next step is to split this data into training and test sets by using Scikit-Learn's built-in train_test_split() method.

# In[29]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


#  **4.Training the Algorithm**
# 
# We have split our data into training and testing sets, and now we will train our algorithm. 

# from sklearn.linear_model import LinearRegression  
# regressor = LinearRegression()  
# regressor.fit(X_train, y_train) 
# 
# print("Training complete.")

# **5. Visualising the model**
# 
# In this step we visualise the trained data

# In[14]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# **6. Making prediciton**

# In[15]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[16]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[28]:


hours = 9.25
own_pred = regressor.predict([[hours]])
print("Predicted Score = {}".format(own_pred[0]))


# **7. Evaluating the model**
# 
# In the last step we are evaluating the by calculating mean absolute error

# In[26]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# **Conclusion:**
# If a student studies for 9.25 hours he/she would likely to score 93.691
