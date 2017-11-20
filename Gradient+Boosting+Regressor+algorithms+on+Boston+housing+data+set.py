
# coding: utf-8

# # This Notebook explains the implementation of Gradient Boosting Algorithm

# In[54]:


# The model that we are going to build this notebook will predict the price of a house when given features of the house.
# As the target variable(price of the house) is continous we are going to use Gradient Boosting Regressr Model.
# importing different packages needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[55]:


#loading the data sets from sklearn package
Boston = datasets.load_boston()

#loading the data set into a data frame
df = pd.DataFrame(Boston.data, columns = Boston.feature_names)
df['H_Val'] = pd.DataFrame(Boston.target)


# In[56]:


#diving the data into Predictor Variables(X) and Target Variable(Y)
X = df.drop(['H_Val'], axis = 1) #dropping the target variable
y = df['H_Val']

#Splitting the dataset into two parts(1.Training Data Set  2.Testing Dataset) using train_test_split( function)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 13)


# In[57]:


#fitting a gradient tree boosting regression model with use 500 trees, with each tree having a depth of 4 levels 
params = {'n_estimators': 500, 'max_depth' : 4, 'learning_rate': 0.01, 'loss': 'ls'}
gbm = GradientBoostingRegressor(**params).fit(X_train, y_train)

#testing out model accuracy by calculating mean squared error value using mean_squared_error() function
y_pred = gbm.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print ("MSE : %.4f"  % mse)


# In[62]:


#computing test set deviance

test_score = np.zeros((params['n_estimators'],), dtype = np.float64)

for i, y_pred in enumerate(gbm.staged_predict(X_test)):
    test_score[i] = gbm.loss_(y_test, y_pred)
    
#Plotting the behavior of algorithm over training and testing error

plt.figure(figsize = (12, 6))
plt.subplot(1,1,1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) +1, gbm.train_score_, 'b_', label = 'Training Set Deviance')
plt.plot(np.arange(params['n_estimators'])+ 1, test_score, 'r_', label = 'Test Set Deviance')

plt.legend(loc = 'upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

