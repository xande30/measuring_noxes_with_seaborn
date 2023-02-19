#!/usr/bin/env python
# coding: utf-8

# In[44]:


get_ipython().run_line_magic('matplotlib', 'inline')

from tabulate import tabulate
from colorama import Fore,Back,Style
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
from colorama import Fore, Back,Style
from sklearn.linear_model import LinearRegression


# In[2]:


get_ipython().run_line_magic('time', '')
boston_dataset = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(boston_dataset, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
boston_dataset = load_boston()
print(boston_dataset.data.shape)
data = pd.DataFrame(boston_dataset,columns=boston_dataset.feature_names)
boston_dataset.feature_names


# In[3]:


boston_dataset.target


# In[4]:


data = pd.DataFrame(data=boston_dataset.data,columns=boston_dataset.feature_names)
data['PRICE'] = boston_dataset.target


# In[5]:


data.head()
data.count()


# In[6]:


print(pd.isnull(data).any())


# In[7]:


print(data.info())


# In[8]:


plt.figure(figsize=(10,6))
sns.distplot(data['PRICE'], bins=50, kde=False,hist=True, color='#03a9f4')
plt.show()


# In[9]:


data['PRICE'].min()
data['PRICE'].max()
data.min()
data.max()
data.describe()   
data["PRICE"].corr(data["RM"])
data["PRICE"].corr(data["PTRATIO"])    


# In[10]:


mask = np.zeros_like(data.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True
print(mask)


# In[11]:


plt.figure(figsize=(20,20))
sns.heatmap(data.corr(),fmt=".1f",cmap="crest",annot=True,linecolor="green", linewidth=.9,annot_kws={"size":14})
sns.set_style('white')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[12]:


import seaborn as sns
nox_dis_corr = round(data['NOX'].corr(data["DIS"]), 3)
plt.scatter(x=data['DIS'], y=data['NOX'], alpha=0.6, s=79, color='seagreen')
plt.title(f'DIS vs NOX (Correlation {nox_dis_corr})', fontsize=12)
plt.xlabel('Dis - Distance from employment', fontsize=12)
plt.ylabel('Nox - Nitric Oxide Pollution', fontsize=12)
sns.set_style("darkgrid")
sns.set_context("talk")
sns.jointplot(x=data['DIS'], y=data['NOX'],kind='hex', alpha=0.9, height=7, color='#76ff03')
plt.show


# In[13]:


rm_tgt_corr = round(data['RM'].corr(data['PRICE']), 3)
plt.figure(figsize=(9,6))
plt.scatter(x=data['RM'], y=data['PRICE'], alpha=0.6, s=79, color='skyblue')
plt.title(f'RM vs PRICE (Correlation {rm_tgt_corr})', fontsize=12)
plt.xlabel('RM - Median Nr. of rooms', fontsize=12)
plt.ylabel('PRICE - property price in 000s', fontsize=12)
sns.set_style("darkgrid")
sns.set_context("talk")
sns.lmplot(x='RM', y='PRICE', data=data, height=7)


# In[ ]:


get_ipython().run_cell_magic('time', '', "sns.pairplot(data, kind='reg',plot_kws={'line_kws':{'color':'cyan'}})\nplt.show()")


# # Training and Test Dataset Split

# In[19]:


prices = data['PRICE']
features = data.drop('PRICE', axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(features, prices, test_size=0.2,random_state=10)
len(X_train) / len(features)


# In[20]:


X_test.shape[0] /features.shape[0]


# In[23]:


regr = LinearRegression()
print(regr.fit(X_train, Y_train))

print('Intercept', regr.intercept_)

pd.DataFrame(data=regr.coef_, index=X_train.columns,columns=['coef'] )


# In[22]:


print('Training data r-squared:', regr.score(X_train, Y_train))
print('Test data r-squared:', regr.score(X_test, Y_test))


# # Model Evaluation By Transformation Of The Data

# In[24]:


data['PRICE'].skew()


# In[25]:


y_log = np.log(data['PRICE'])
y_log.tail


# In[26]:


y_log.skew()


# In[28]:


sns.displot(y_log)
plt.title(f'Log price with skew {y_log.skew()}')
plt.show()


# In[35]:


sns.lmplot(x='LSTAT', y='PRICE',data=data, height=7,scatter_kws={'alpha':0.6}, line_kws={'color':'darkred'})
plt.show()


# In[38]:


transformed_data = features
transformed_data['LOG_PRICE'] = y_log
sns.lmplot(x='LSTAT', y='LOG_PRICE',data=transformed_data, height=7,scatter_kws={'alpha':0.6}, line_kws={'color':'green'})
plt.show()


# # Regression using log prices

# In[41]:


prices = np.log(data['PRICE']) # use log prices
features = data.drop('PRICE', axis=1)
x_train, x_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)
regr = LinearRegression()
regr.fit(x_train, y_train)
print('Training data r-squaredP:', regr.score(x_train, y_train))
print('Test data r-squared:', regr.score(x_test, y_test))
print('Intercept', regr.intercept_)
pd.DataFrame(data=regr.coef_,index=x_train.columns, columns=['coef'])


# In[42]:


#premium property payment next to the river
np.e**0.080475


# In[51]:


# Evaluating Coefficients
X_incl_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const)
results = model.fit()
results.params

results.pvalues
pd.DataFrame({'coef': results.params,'p-value':round(results.pvalues, 3)})


# # Testing for Multicollinearity
# # $$ VIF_{TAX} = \frac{1}{(1-R _{TAX} ^ 2)} $$

# In[53]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[56]:


variance_inflation_factor(exog=X_incl_const.values, exog_idx=1)
#type(X_incl_const)


# In[63]:


print(len(X_incl_const.columns))
X_incl_const.shape[1]


# In[72]:


for i in range (X_incl_const.shape[1]):
    print(variance_inflation_factor(exog=X_incl_const.values, exog_idx=i))
    print('Áll Ok')


# In[78]:


vif = []

for i in range (X_incl_const.shape[1]):
    vif.append(variance_inflation_factor(exog=X_incl_const.values, exog_idx=i))
print(vif)
pd.DataFrame({'coef_name': X_incl_const.columns, 'vif':np.around(vif, 2)})


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




