# JUPITER NOTEBOOK

# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


import math



# In[2]:


X_raw = np.loadtxt('hw1-q1x.csv')
y_raw = np.loadtxt('hw1-q1y.csv')


# In[3]:


train_set, test_set = train_test_split(X_raw, test_size=0.2, random_state=42)
y_train, y_test = train_test_split(y_raw, test_size=0.2, random_state=42)


# In[4]:


X_train = pd.DataFrame(train_set)
X_test = pd.DataFrame(test_set)


# In[ ]:


# Part (B)


# In[5]:


reg_parameter = [0, 0.1] + [10**i for i in range(6)]
print(reg_parameter)


# In[11]:


rmse = []
test_rmse = []
weight_vectors = []
l2_norm = []

for reg in reg_parameter:
    lin_reg = Ridge(alpha=reg, fit_intercept=True)
    lin_reg.fit(X_train, y_train)

    y_predict = lin_reg.predict(X_train)
    lin_mse = mean_squared_error(y_train, y_predict)
    lin_rmse = np.sqrt(lin_mse)
    rmse.append(lin_rmse)

    weight_vectors.append(lin_reg.coef_)
    l2_norm.append(np.linalg.norm(lin_reg.coef_))

    test_y_predict = lin_reg.predict(X_test)
    lin_mse_test = mean_squared_error(y_test, test_y_predict)
    lin_rmse_test = np.sqrt(lin_mse_test)
    test_rmse.append(lin_rmse_test)
print(rmse)
print(test_rmse)
print(l2_norm)


# In[12]:


plt.plot(reg_parameter, rmse, 'b', label='Training data')
plt.plot(reg_parameter, test_rmse, 'g', label='Test data')
plt.xlabel('Regularization parameter')
plt.ylabel('Linear Regression RMSE')
plt.xscale('log')
plt.legend()
plt.show()


# In[10]:


plt.plot(reg_parameter, l2_norm)
plt.xlabel('Regularization parameter')
plt.ylabel('L2 Norm of Weight Vectors')
plt.xscale('log')
plt.show()


# In[11]:

#
#weight_list = []
#print(weight_vectors)
#for i in range(2):
#    y_coord = [weight_vectors[j][0][i] for j in range(8)]
#    weight_list.append(y_coord)

#for i in range(2):
#    plt.plot(reg_parameter, weight_list[i], label='Weight # %s' % (i))
#plt.xscale('log')
#plt.xlabel('Regularization parameter')
#plt.ylabel('Actual values of the weights')
#plt.show()


# In[ ]:


# Part (C)


# In[12]:



#for par in reg_parameter:
#    lin_reg = Ridge(alpha=par)
#    scores = cross_val_score(lin_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5)


#    print('For regularization parameter: %.1f' %par)
##    print('Training scores: ', np.sqrt(-scores['train_score']))
#    print('Average training score: ',np.average(np.sqrt(-scores['train_score'])))
#    print('Validation scores: ',np.sqrt(-scores['test_score']))
#    print('Average validation score: ', np.average(np.sqrt(-scores['test_score'])))
#    print('\n')


# In[ ]:


# Part (E)


# In[7]:


list_dataframes = [] # store list of transformed training set dataframes, each one has a different variance
means = [-1, -0.5, 0, 0.5, 1]
variances = [0.1, 0.5, 1, 5]

for var in variances:
    df = pd.DataFrame()
    for mean in means:
        for column in np.array(train_set).T:

            new_col = pd.DataFrame(np.exp(-((column-mean)**2)/(2*var) ) )
            df = pd.concat([df,new_col], axis=1)
    list_dataframes.append(df)
    print(df.shape)

list_test = [] # store list of transformed test set dataframes, each one has a different variance

for var in variances:
    df = pd.DataFrame()
    for mean in means:
        for column in np.array(test_set).T:
            new_col = pd.DataFrame(np.exp(-((column-mean)**2)/(2*var) ) )
            df = pd.concat([df,new_col], axis=1)
    list_test.append(df)
    print(df.shape)


# In[14]:


gauss_train_rmse = []
gauss_test_rmse = []
for train_df, test_df in zip(list_dataframes,list_test):
    new_lin_reg = LinearRegression()
    new_lin_reg.fit(train_df, y_train)

    y_predict_train = new_lin_reg.predict(train_df)
    lin_mse = mean_squared_error(y_train, y_predict_train)
    lin_rmse = np.sqrt(lin_mse)
    gauss_train_rmse.append(lin_rmse)

    y_predict_test = new_lin_reg.predict(test_df)
    lin_mse = mean_squared_error(y_test, y_predict_test)
    lin_rmse = np.sqrt(lin_mse)
    gauss_test_rmse.append(lin_rmse)

print(gauss_train_rmse)
print(gauss_test_rmse)


# In[22]:


plt.plot(variances, gauss_test_rmse, label='Testing error')
plt.plot(variances, gauss_train_rmse, label='Training error')
for scalar in rmse:
    plt.axhline(y=scalar, linestyle='-', color='g')
for scalar in test_rmse:
    plt.axhline(y=scalar, linestyle='-', color='r')
plt.xlabel('Variance')
plt.ylabel('Errors')
plt.legend()
plt.show()

