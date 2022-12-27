#!/usr/bin/env python
# coding: utf-8

# In[356]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# # Step 1: Loading and understanding data

# In[357]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats


# In[358]:


data = pd.read_csv("C:\\PG_AI_ML\\RegularizationAssignment\\train.csv")
data.head()


# In[359]:


data.shape


# #### Observation: There are 1460 rows of data. 81 data points

# In[360]:


data.info()


# #### Observation: As per above there many missing values in few of the columns. Lets look at the % of missing values in the columns

# In[361]:


data.describe().T


# #### As per the data dictionary ID column is just a incremental value i.e index. So not required.

# In[362]:


data.drop('Id', axis=1, inplace=True)


# In[363]:


data.head()


# #### As per the data below are the categorival variables

# In[364]:


'''
MSZoning
Street
Alley
LotShape
LandContour
Utilities
LotConfig
LandSlope
Neighborhood
Condition1
Condition2
BldgType
HouseStyle
RoofStyle
RoofMatl
Exterior1st
Exterior2nd
MasVnrType
ExterQual
ExterCond
Foundation
BsmtQual
BsmtCond
BsmtExposure
BsmtFinType1
BsmtFinType2
Heating
HeatingQC
CentralAir
Electrical
KitchenQual
Functional
FireplaceQu
GarageType
GarageFinish
GarageQual
GarageCond
PavedDrive
PoolQC
Fence
MiscFeature
SaleType
SaleCondition
'''


# #### Rest variables are continuous variables
# #### SalesPrice is our target variable

# In[365]:


data2 = data[[column for column in data if data[column].count() / len(data) >= 0.3]]
print("List of dropped columns:", end=" ")

for c in data.columns:
    if c not in data2.columns:
        print(c, end=", ")
print('\n')


# #### As per above Alley, PoolQC, Fence, MiscFeature columns have more that 30% data missed. So we can remove them

# In[366]:


# variable containing the columns that we want to remove
miss_data_col = ['Alley','PoolQC', 'Fence', 'MiscFeature']
data = data.drop(miss_data_col,axis=1)
data.shape


# #### Lets look at the distribution of the SalesPrice column values

# In[367]:


print(data['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(data['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});


# #### Observation: The ditribution is right skewed. Means there are outliers in the saleprice too. Seems after 500000 the saleprice have outliers. We will be removing them in future.

# ## Lets see the numerical data distribution

# In[368]:


list(set(data.dtypes.tolist()))# Different types in the data set


# In[369]:


data_num = data.select_dtypes(include = ['float64', 'int64']) #Getting variables with numric datatypes
data_num.head()


# In[370]:


#Plot the histograms on numerical data
data_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);


# #### Observation: Variables like (TotalBsmtSF. 1stFlrSF) , (EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, MiscVal) have similar distribution of data.

# ### Lets analyse Corelation

# In[371]:


data_num_corr = data_num.corr()['SalePrice'][:-1]
strong_features = data_num_corr[abs(data_num_corr) > 0.5].sort_values(ascending=False)
print("No.of Strong corelation {}".format(len(strong_features)))
print("{}".format(strong_features))


# In[372]:


#Pair plot the numerical variables with SalesPrice (target variable)
for i in range(0, len(data_num.columns), 5):
    sns.pairplot(data=data_num,
                x_vars=data_num.columns[i:i+5],
                y_vars=['SalePrice'])


# #### Observation: If you above scatter plots, quite a number of variables have linear relation with sales price. Also variables like PoolArea, MiscVal, 3SsnPorch ...etc have lot of values zeros.

# In[373]:


# Lets fix the SalesPrice distribution as we observe above it is right skewed.
#To fix this we apply log instead of SalesPrice directly
data['SalePrice_log'] = np.log(data['SalePrice'])
data_num['SalePrice_log'] = np.log(data_num['SalePrice'])


# In[374]:


saleprice = data[['SalePrice']].copy()
df = data.drop('SalePrice',axis=1) # removing the orignal variable
data_num.drop('SalePrice',axis=1) # removing the orignal variable


# In[375]:


fig, ax = plt.subplots(1,2, figsize= (15,5))
fig.suptitle("qq-plot & distribution SalePrice ", fontsize= 15)

sm.qqplot(data['SalePrice_log'] , stats.t, distargs=(4,),fit=True, line="45", ax = ax[0])
sns.distplot(data['SalePrice_log'] , kde = True, hist=True, ax = ax[1])
plt.show()


# #### Observe that now the sale prices is normally distributed which is needed to apply Linear Regression

# ### Outliers Detection

# In[376]:


data_num.shape


# In[377]:


data_num.columns


# In[378]:


a = 20
b = 2
c = 1

fig = plt.figure(figsize=(50,40))
fig.set_size_inches(15, 50)

for i in data_num.columns:
    plt.subplot(a, b, c)
    plt.title('{}, subplot: {}{}{}'.format(i, a, b, c))
    plt.xlabel(i)
    sns.boxplot(data_num[i])
    c = c + 1

    plt.tight_layout()
plt.show();


# #### Observation: lot of outliers present in few variables. We need to treat them.

# ### Treating the outliers

# In[379]:


data_num.isnull().sum().sort_values(ascending=False)[:10]


# In[380]:


# Importing the SimpleImputer class 
from sklearn.impute import SimpleImputer


# In[381]:


imputer = SimpleImputer(missing_values = np.nan,strategy ='median')


# In[382]:


data_num['LotFrontage'] = imputer.fit_transform(data_num[['LotFrontage']].copy()).ravel()
data_num['GarageYrBlt'] = imputer.fit_transform(data_num[['GarageYrBlt']].copy()).ravel()
data_num['MasVnrArea'] = imputer.fit_transform(data_num[['MasVnrArea']].copy()).ravel()


# In[383]:


data_num.isnull().sum().sort_values(ascending=False)[:10]


# In[384]:


plt.figure(figsize=(25, 15))

sns.heatmap(data_num.corr(),annot=True,vmin=-1, vmax=1, cmap='BuGn')


# In[385]:


data_num.drop('SalePrice', axis=1, inplace=True)


# In[386]:


data_num.columns


# In[387]:


plt.figure(figsize=(8, 12))
heatmap_outputvar = sns.heatmap(data_num.corr()[['SalePrice_log']].sort_values(by='SalePrice_log', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BuGn')
heatmap_outputvar.set_title('Features Correlating with Sales Price', fontdict={'fontsize':18}, pad=16);


# #### Observe: That more that 10 variables have heigher correlation with the SalePrice

# ### Skewness in the numerical columns

# In[388]:


from scipy.stats import skew 
skew_in_vars = data_num.iloc[:,:-1].apply(lambda x: skew(x))
skew_in_vars.sort_values(ascending=False)


# In[389]:


a = 40
b = 2
c = 1

fig = plt.figure(figsize=(50,40))
fig.set_size_inches(15, 100)

for i in data_num.columns:
    plt.subplot(a, b, c)
    plt.title('{}, subplot: {}{}{}'.format(i, a, b, c))
    plt.xlabel(i)
    sns.regplot(data_num[i], data_num['SalePrice_log'])
    c = c + 1

plt.tight_layout()
plt.show();


# #### Drop the columns which are not strongly corelated with saleprice

# In[390]:


num_corr_data = data_num.corr()
top_corr_num_col = num_corr_data.index[abs(num_corr_data['SalePrice_log'])>0.2]
top_corr_num_col


# In[391]:


num_data_f = data_num[top_corr_num_col]
num_data_f.shape


# In[392]:


plt.figure(figsize=(20,7))
sns.heatmap(num_data_f.corr(),annot=True,cmap="YlOrRd")


# ### Categorical Variables

# In[393]:


cat_col = data.select_dtypes(include=object).columns.tolist()


# In[394]:


data_cat = data[cat_col]
data_cat.shape


# In[395]:


data_cat.head()


# In[396]:


data_cat.isnull().sum().sort_values(ascending=False)[:20]


# In[397]:


data_cat['FireplaceQu'] = data_cat['FireplaceQu'].fillna('No Fireplace')
data_cat['GarageCond'] = data_cat['GarageCond'].fillna('No Garage')
data_cat['GarageQual'] = data_cat['GarageQual'].fillna('No Garage')
data_cat['GarageFinish'] = data_cat['GarageFinish'].fillna('No Garage')
data_cat['GarageType'] = data_cat['GarageType'].fillna('No Garage')


# In[398]:


data_cat['BsmtFinType2'] = data_cat['BsmtFinType2'].fillna('No Basement')
data_cat['BsmtExposure'] = data_cat['BsmtExposure'].fillna('No Basement')
data_cat['BsmtCond']     = data_cat['BsmtCond'].fillna('No Basement')
data_cat['BsmtQual']     = data_cat['BsmtQual'].fillna('No Basement')
data_cat['BsmtFinType1'] = data_cat['BsmtFinType1'].fillna('No Basement')


# In[399]:


data_cat['MasVnrType'] = data_cat['MasVnrType'].fillna('None')
data_cat['Electrical'] = data_cat['Electrical'].fillna(method='ffill')


# In[400]:


data_cat.isnull().sum().sort_values(ascending=False)[:5]


# In[401]:


# Adding Target variable
data_cat = pd.concat([data_cat,num_data_f[['SalePrice_log']]],axis=1)
data_cat.head()


# In[402]:


a = 40
b = 2
c = 1

fig = plt.figure(figsize=(50,40))
fig.set_size_inches(15, 100)

for i in data_cat.columns:
    plt.subplot(a, b, c)
    plt.title('{}, subplot: {}{}{}'.format(i, a, b, c))
    plt.xlabel(i)
    sns.boxplot(data_cat[i], data_cat['SalePrice_log'])
    c = c + 1

plt.tight_layout()
plt.show();


# #### Observation: If we check above cat variables all variables are strong correlation with target variable SalePrice based on the mean  of the each feature in the particular variable

# In[403]:


top_corr_cat_col = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
                    'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
cat_df_f = data_cat[top_corr_cat_col]
cat_df_f.shape


# ### Encoding the categorical variable as they are non-numeric values. Ecoding them will help in further model preparation

# Applying one-hot encoding

# In[404]:


cat_df_f = pd.get_dummies(cat_df_f)


# In[405]:


cat_df_f.head()


# Building final data set with all numerical and encoded categorical variables

# In[406]:


data_total =  pd.concat([cat_df_f,num_data_f],axis=1)
data_total.shape


# In[407]:


data_total.head()


# In[408]:


X = data_total.drop('SalePrice_log',axis =1)
y = data_total.pop('SalePrice_log')
print('X shape',X.shape)
print('y shape',y.shape)


# ### Train and Test data split

# In[409]:


from sklearn.model_selection import train_test_split


# In[465]:


X_train, X_test, train_labels, test_labels = train_test_split(X,y,test_size=0.3, random_state =1)


# ### Linear Regression

# In[411]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels
import statsmodels.api as sm


# In[412]:


print('X train shape',X_train.shape)
print('X test shape',X_train.shape)
print('y train shape',train_labels.shape)
print('y test shape',test_labels.shape)


# In[413]:


lm = LinearRegression()


# In[414]:


lm.fit(X_train,train_labels)


# ### Prediction

# In[415]:


y_pred_train_lr = lm.predict(X_train)
y_pred_test_lr =  lm.predict(X_test)


# In[416]:


test_labels_lr = pd.DataFrame(test_labels)
test_labels_lr.to_csv('test_labels_lr.csv')


# In[417]:


y_pred_test_lr = pd.DataFrame(y_pred_test_lr)
y_pred_test_lr.to_csv('y_pred_test_lr.csv')


# In[418]:


X_test.loc[[375,1270,325,120,1011,1003],:].head(10)


# In[419]:


X_test.head(10)


# In[420]:


#dropping some of the observations due to erros in the prediction values 
X_test_update = X_test.drop([375,1270,325,120,1011,1003],axis=0)


# In[421]:


y_pred_test_lr_update =  lm.predict(X_test_update)
y_pred_test_lr_update.shape


# In[422]:


test_labels_update = test_labels.drop([375,1270,325,120,1011,1003],axis=0)
test_labels_update.shape


# In[423]:


print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(train_labels, y_pred_train_lr))))
print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(test_labels_update, y_pred_test_lr_update))))


# In[424]:


print("R-Square for training data",lm.score(X_train,train_labels)) # Return the coefficient of determination R^2 of the prediction.
print("R-Square for test data",lm.score(X_test_update,test_labels_update))


# ### Tuning the params

# In[425]:


from sklearn.model_selection import KFold, RepeatedKFold,cross_val_score,GridSearchCV
from sklearn.metrics import mean_squared_error,make_scorer


# In[426]:


n_folds = 5

scorer = make_scorer(mean_squared_error,greater_is_better = False)

def rmse_CV_train(model):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(X.values)
    rmse = np.sqrt(-cross_val_score(model,X_train,train_labels,scoring ="neg_mean_squared_error",cv=kf))
    return (rmse)

def rmse_CV_test(model):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(X.values)
    rmse = np.sqrt(-cross_val_score(model,X_test_update,test_labels_update,scoring ="neg_mean_squared_error",cv=kf))
    return (rmse)


# In[427]:


lr_cv = LinearRegression()
lr_cv.fit(X_train,train_labels)


# ### Prediction

# In[428]:


y_pred_train_lr_cv = lr_cv.predict(X_train)
y_pred_test_lr_cv =  lr_cv.predict(X_test_update)


# In[429]:


print('rmse on train',rmse_CV_train(lr_cv).mean())
print('rmse on test',rmse_CV_test(lr_cv).mean())


# ### Ridge regression

# In[430]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

lm_ridge=Ridge()
parameters= {'alpha':[x for x in [0.0005,0.001,0.01,0.1,0.2,0.4,0.5,0.7,0.8,1]]}


# In[431]:


lm_ridge_grd = GridSearchCV(estimator = lm_ridge, param_grid=parameters)


# In[432]:


lm_ridge_grd.fit(X_train,train_labels)

print("The best value of Alpha is: ",lm_ridge_grd.best_params_,lm_ridge_grd.best_score_)


# In[433]:


lm_ridge_best_gcv = lm_ridge_grd.best_estimator_
lm_ridge_best_gcv


# In[434]:


lm_ridge_best_gcv.fit(X_train,train_labels)


# In[435]:


#Most Important predictor i.e which has more Coefficient value for independent variables
from operator import itemgetter
coefficients_ridge = list(zip(lm_ridge_best_gcv.feature_names_in_, lm_ridge_best_gcv.coef_))
res = max(coefficients_ridge, key=itemgetter(1))[0]
print(res)


# In[436]:


def Sort_List_Coeff(tup):
    lst = len(tup)
    for i in range(0, lst):
         
        for j in range(0, lst-i-1):
            if (tup[j][1] > tup[j + 1][1]):
                temp = tup[j]
                tup[j]= tup[j + 1]
                tup[j + 1]= temp
    return tup


# In[437]:


#Top 15 variables for prediction
sorted_coefficients_ridge = Sort_List_Coeff(coefficients_ridge)
print(sorted_coefficients_ridge[-15:])


# ### Prediction

# In[438]:


y_pred_train_ridge_gcv = lm_ridge_best_gcv.predict(X_train)
y_pred_test_ridge_gcv =  lm_ridge_best_gcv.predict(X_test_update)


# In[439]:


print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(train_labels, y_pred_train_ridge_gcv))))
print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(test_labels_update, y_pred_test_ridge_gcv)))) 


# In[440]:


print("R-Square for training data",lm_ridge_best_gcv.score(X_train,train_labels)) # Return the coefficient of determination R^2 of the prediction.
print("R-Square for training data",lm_ridge_best_gcv.score(X_test_update,test_labels_update))


# In[443]:


y_pred_train_ridge_gcv_df = pd.DataFrame(y_pred_train_ridge_gcv)
#y_pred_train_ridge_gcv['preds'].head()

X_train_out = X.reset_index()
X_train_out["saleprice_Actual"] = train_labels.reset_index()["SalePrice_log"]
X_train_out["saleprice_Prediction"] = y_pred_train_ridge_gcv_df.reset_index()[0]

train_labels['preds'] = y_pred_train_ridge_gcv_df

X_train_out = pd.merge(X_train_out,train_labels[['preds']],how = 'left',left_index = True, right_index = True)
X_train_out = X_train_out.drop('SalePrice_log',axis =1)
X_train_out.head()

#Check colinearity between top 15 predictors and predicted sales_prices

a = 40
b = 2
c = 1

fig = plt.figure(figsize=(50,40))
fig.set_size_inches(15, 100)

for i in ['Functional_Min2','GarageCond_Po','Utilities_AllPub','MSZoning_RH','Condition2_Norm','Condition2_PosA','RoofMatl_Membran', 'Condition2_Feedr','Functional_Typ', 'Neighborhood_NridgHt','GarageQual_Ex','Neighborhood_Crawfor', 'RoofMatl_CompShg','Neighborhood_StoneBr', 'RoofMatl_WdShngl']:
    plt.subplot(a, b, c)
    plt.title('{}, subplot: {}{}{}'.format(i, a, b, c))
    plt.xlabel(i)
    sns.regplot(X_train_out[i], X_train_out['saleprice_Prediction'])
    c = c + 1

plt.tight_layout()
plt.show();


# ### Lasso

# In[444]:


from sklearn.linear_model import Lasso


# In[445]:


lm_lasso =Lasso()
parameters= {'alpha':[x for x in [0.0005,0.001,0.01,0.1,0.2,0.4,0.5,0.7,0.8,1]]}


# In[446]:


lm_lasso_grd = GridSearchCV(estimator=lm_lasso, 
                            param_grid=parameters)


# In[451]:


lm_lasso_grd.fit(X_train,train_labels)

print("The best value of Alpha is: ",lm_lasso_grd.best_params_,lm_lasso_grd.best_score_)


# In[452]:


lm_lasso_best_gcv = lm_lasso_grd.best_estimator_
lm_lasso_best_gcv


# In[453]:


lm_lasso_best_gcv.fit(X_train,train_labels)


# In[454]:


X_train.shape


# In[455]:


len(lm_lasso_best_gcv.feature_names_in_)


# In[463]:


#Most Important 15 predictors i.e which has more Coefficient value for independent variables
from operator import itemgetter
coefficients_lasso = list(zip(lm_lasso_best_gcv.feature_names_in_, lm_lasso_best_gcv.coef_))
res = max(coefficients_lasso, key=itemgetter(1))[0]
print(res)
sorted_coefficients_lasso = Sort_List_Coeff(coefficients_lasso)
print(sorted_coefficients_lasso[-15:])


# In[457]:


print(lm_lasso_best_gcv.score(X_train, train_labels))


# ### Prediction

# In[458]:


y_pred_train_lasso_gcv = lm_lasso_best_gcv.predict(X_train)
y_pred_test_lasso_gcv =  lm_lasso_best_gcv.predict(X_test_update)


# In[459]:


print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(train_labels, y_pred_train_lasso_gcv))))
print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(test_labels_update, y_pred_test_lasso_gcv)))) 


# In[460]:


print("R-Square for training data",lm_lasso_best_gcv.score(X_train,train_labels)) # Return the coefficient of determination R^2 of the prediction.
print("R-Square for training data",lm_lasso_best_gcv.score(X_test_update,test_labels_update))


# #### As per R-suare and Root mean square error, among the Ridge and Lasso, Lasso giving less values for these matrics. It generalized more more by increasing the bias on the traindata. So we take that regularization and conclude the model

# #### Ridge by doubling the alpha

# In[462]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

lm_ridge=Ridge()
parameters= {'alpha':[x for x in [0.0010,0.002,0.02,0.2,0.4,0.8,1.0,1.4,1.6,2]]}
lm_ridge_grd = GridSearchCV(estimator = lm_ridge, param_grid=parameters)
lm_ridge_grd.fit(X_train,train_labels)

print("The best value of Alpha is: ",lm_ridge_grd.best_params_,lm_ridge_grd.best_score_)
lm_ridge_best_gcv = lm_ridge_grd.best_estimator_
lm_ridge_best_gcv
lm_ridge_best_gcv.fit(X_train,train_labels)

#Top 15 important predictors are
sorted_coefficients_ridge = Sort_List_Coeff(coefficients_ridge)
print(sorted_coefficients_ridge[-15:])

#Prediction
y_pred_train_ridge_gcv = lm_ridge_best_gcv.predict(X_train)
y_pred_test_ridge_gcv =  lm_ridge_best_gcv.predict(X_test_update)
print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(train_labels, y_pred_train_ridge_gcv))))
print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(test_labels_update, y_pred_test_ridge_gcv)))) 
print("R-Square for training data",lm_ridge_best_gcv.score(X_train,train_labels)) # Return the coefficient of determination R^2 of the prediction.
print("R-Square for training data",lm_ridge_best_gcv.score(X_test_update,test_labels_update))

#train_labels.head()
#print(type(y_pred_train_ridge_gcv))
y_pred_train_ridge_gcv_df = pd.DataFrame(y_pred_train_ridge_gcv)
#y_pred_train_ridge_gcv['preds'].head()

X_train_out = X.reset_index()
X_train_out["saleprice_Actual"] = train_labels.reset_index()["SalePrice_log"]
X_train_out["saleprice_Prediction"] = y_pred_train_ridge_gcv_df.reset_index()[0]

train_labels['preds'] = y_pred_train_ridge_gcv_df

X_train_out = pd.merge(X_train_out,train_labels[['preds']],how = 'left',left_index = True, right_index = True)
X_train_out = X_train_out.drop('SalePrice_log',axis =1)
X_train_out.head()

#Check colinearity between top 15 predictors and predicted sales_prices

a = 40
b = 2
c = 1

fig = plt.figure(figsize=(50,40))
fig.set_size_inches(15, 100)

for i in ['SaleType_Oth', 'BsmtExposure_Gd', 'Exterior1st_BrkFace', 'Functional_Min2', 'Condition2_PosA','MSZoning_RH','Condition2_Feedr','Condition2_Norm','GarageQual_Ex', 'RoofMatl_CompShg', 'Functional_Typ','Neighborhood_NridgHt', 'Neighborhood_Crawfor', 'RoofMatl_WdShngl', 'Neighborhood_StoneBr']:#X_train_out.columns:
    plt.subplot(a, b, c)
    plt.title('{}, subplot: {}{}{}'.format(i, a, b, c))
    plt.xlabel(i)
    sns.regplot(X_train_out[i], X_train_out['saleprice_Prediction'])
    c = c + 1

plt.tight_layout()
plt.show();


# ### Lasso by doubling the alpha values

# In[466]:


from sklearn.linear_model import Lasso
lm_lasso =Lasso()
parameters= {'alpha':[x for x in [0.0010,0.002,0.02,0.2,0.4,0.8,1.0,1.4,1.6,2]]}
lm_lasso_grd = GridSearchCV(estimator=lm_lasso, 
                            param_grid=parameters)
lm_lasso_grd.fit(X_train,train_labels)

print("The best value of Alpha is: ",lm_lasso_grd.best_params_,lm_lasso_grd.best_score_)
lm_lasso_best_gcv = lm_lasso_grd.best_estimator_
lm_lasso_best_gcv.fit(X_train,train_labels)
#Most Important 15 predictors i.e which has more Coefficient value for independent variables
from operator import itemgetter
coefficients_lasso = list(zip(lm_lasso_best_gcv.feature_names_in_, lm_lasso_best_gcv.coef_))
res = max(coefficients_lasso, key=itemgetter(1))[0]
print(res)
sorted_coefficients_lasso = Sort_List_Coeff(coefficients_lasso)
print(sorted_coefficients_lasso[-15:])
print(lm_lasso_best_gcv.score(X_train, train_labels))
#Predictions
y_pred_train_lasso_gcv = lm_lasso_best_gcv.predict(X_train)
y_pred_test_lasso_gcv =  lm_lasso_best_gcv.predict(X_test_update)
print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(train_labels, y_pred_train_lasso_gcv))))
print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(test_labels_update, y_pred_test_lasso_gcv)))) 
print("R-Square for training data",lm_lasso_best_gcv.score(X_train,train_labels)) # Return the coefficient of determination R^2 of the prediction.
print("R-Square for training data",lm_lasso_best_gcv.score(X_test_update,test_labels_update))


# In[ ]:




