""" 
Automatic selection of best imputation technique with Sklearn
- In this example we will do a grid search over the imputation methods available in Scikit-learn to determine 
        which imputation technique works best for this dataset and the machine learning model of choice.

- We will also train a very simple machine learning model as part of a small pipeline.

- We will use the House Price dataset for the example.
"""

import pandas as pd
import numpy as np
# import classes for imputation
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# import extra classes for modelling
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV

np.random.seed(0)

# load dataset with all the variables

data = pd.read_csv('dataset/house-prices-advanced-regression-techniques/train.csv',)
data.head()
""" 
   Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape LandContour Utilities  ... PoolArea PoolQC Fence MiscFeature MiscVal MoSold YrSold  SaleType  SaleCondition  SalePrice
0   1          60       RL         65.0     8450   Pave   NaN      Reg         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   2008        WD         Normal     208500
1   2          20       RL         80.0     9600   Pave   NaN      Reg         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      5   2007        WD         Normal     181500
2   3          60       RL         68.0    11250   Pave   NaN      IR1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      9   2008        WD         Normal     223500
3   4          70       RL         60.0     9550   Pave   NaN      IR1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   2006        WD        Abnorml     140000
4   5          60       RL         84.0    14260   Pave   NaN      IR1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0     12   2008        WD         Normal     250000

[5 rows x 81 columns] """

# find categorical variables
# those of type 'Object' in the dataset
features_categorical = [c for c in data.columns if data[c].dtypes=='O']

# find numerical variables
# those different from object and also excluding the target SalePrice
features_numerical = [c for c in data.columns if data[c].dtypes!='O' and c !='SalePrice']

# inspect the categorical variables

data[features_categorical].head()
""" 
  MSZoning Street Alley LotShape LandContour Utilities LotConfig LandSlope Neighborhood Condition1  ... GarageType GarageFinish GarageQual GarageCond PavedDrive PoolQC Fence MiscFeature SaleType SaleCondition
0       RL   Pave   NaN      Reg         Lvl    AllPub    Inside       Gtl      CollgCr       Norm  ...     Attchd          RFn         TA         TA          Y    NaN   NaN         NaN       WD        Normal
1       RL   Pave   NaN      Reg         Lvl    AllPub       FR2       Gtl      Veenker      Feedr  ...     Attchd          RFn         TA         TA          Y    NaN   NaN         NaN       WD        Normal
2       RL   Pave   NaN      IR1         Lvl    AllPub    Inside       Gtl      CollgCr       Norm  ...     Attchd          RFn         TA         TA          Y    NaN   NaN         NaN       WD        Normal
3       RL   Pave   NaN      IR1         Lvl    AllPub    Corner       Gtl      Crawfor       Norm  ...     Detchd          Unf         TA         TA          Y    NaN   NaN         NaN       WD       Abnorml
4       RL   Pave   NaN      IR1         Lvl    AllPub       FR2       Gtl      NoRidge       Norm  ...     Attchd          RFn         TA         TA          Y    NaN   NaN         NaN       WD        Normal

[5 rows x 43 columns] """
# inspect the numerical variables

data[features_numerical].head()
""" 
   Id  MSSubClass  LotFrontage  LotArea  OverallQual  OverallCond  YearBuilt  YearRemodAdd  MasVnrArea  ...  WoodDeckSF  OpenPorchSF  EnclosedPorch  3SsnPorch  ScreenPorch  PoolArea  MiscVal  MoSold  YrSold
0   1          60         65.0     8450            7            5       2003          2003       196.0  ...           0           61              0          0            0         0        0       2    2008
1   2          20         80.0     9600            6            8       1976          1976         0.0  ...         298            0              0          0            0         0        0       5    2007
2   3          60         68.0    11250            7            5       2001          2002       162.0  ...           0           42              0          0            0         0        0       9    2008
3   4          70         60.0     9550            7            5       1915          1970         0.0  ...           0           35            272          0            0         0        0       2    2006
4   5          60         84.0    14260            8            5       2000          2000       350.0  ...         192           84              0          0            0         0        0      12    2008

[5 rows x 37 columns] """

# separate intro train and test set

X_train, X_test, y_train, y_test = train_test_split(
    data.drop('SalePrice', axis=1),  # just the features
    data['SalePrice'],  # the target
    test_size=0.3,  # the percentage of obs in the test set
    random_state=0)  # for reproducibility

X_train.shape, X_test.shape
# ((1022, 80), (438, 80))

# We create the preprocessing pipelines for both
# numerical and categorical data

# adapted from Scikit-learn code available here under BSD3 license:
# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numeric_transformer, features_numerical),
        ('categorical', categorical_transformer, features_categorical)])

# Note that to initialise the pipeline I pass any argument to the transformers.
# Those will be changed during the gridsearch below.
# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', Lasso(max_iter=2000))])

# now we create the grid with all the parameters that we would like to test
param_grid = {
    'preprocessor__numerical__imputer__strategy': ['mean', 'median'],
    'preprocessor__categorical__imputer__strategy': ['most_frequent', 'constant'],
    'classifier__alpha': [10, 100, 200],
}

grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, scoring='r2')

# cv=3 is the cross-validation
# no_jobs =-1 indicates to use all available cpus
# scoring='r2' indicates to evaluate using the r squared

# for more details in the grid parameters visit:
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
""" 
- When setting the grid parameters, this is how we indicate the parameters:

preprocessor__numerical__imputer__strategy': ['mean', 'median'],

- the above line of code indicates that I would like to test the mean and the median in the imputer step of 
        the numerical processor.

preprocessor__categorical__imputer__strategy': ['most_frequent', 'constant']

- the above line of code indicates that I would like to test the most frequent or a constant value in the imputer 
        step of the categorical processor

classifier__alpha': [0.1, 1.0, 0.5]

- the above line of code indicates that I want to test those 3 values for the alpha parameter of Lasso. 
        Note that Lasso is the 'classifier' step of our last pipeline
 """
# and now we train over all the possible combinations of the parameters above
grid_search.fit(X_train, y_train)

# and we print the best score over the train set
print(("best linear regression from grid search: %.3f"% grid_search.score(X_train, y_train)))
# best linear regression from grid search: 0.933

# we can print the best estimator parameters like this
grid_search.best_estimator_
""" 
Pipeline(steps=[('preprocessor',
                 ColumnTransformer(transformers=[('numerical',
                                                  Pipeline(steps=[('imputer',
                                                                   SimpleImputer(strategy='median')),
                                                                  ('scaler',
                                                                   StandardScaler())]),
                                                  ['Id', 'MSSubClass',
                                                   'LotFrontage', 'LotArea',
                                                   'OverallQual', 'OverallCond',
                                                   'YearBuilt', 'YearRemodAdd',
                                                   'MasVnrArea', 'BsmtFinSF1',
                                                   'BsmtFinSF2', 'BsmtUnfSF',
                                                   'TotalBsmtSF', '1stFlrSF',
                                                   '2ndFlrS...
                                                   'LotConfig', 'LandSlope',
                                                   'Neighborhood', 'Condition1',
                                                   'Condition2', 'BldgType',
                                                   'HouseStyle', 'RoofStyle',
                                                   'RoofMatl', 'Exterior1st',
                                                   'Exterior2nd', 'MasVnrType',
                                                   'ExterQual', 'ExterCond',
                                                   'Foundation', 'BsmtQual',
                                                   'BsmtCond', 'BsmtExposure',
                                                   'BsmtFinType1',
                                                   'BsmtFinType2', 'Heating',
                                                   'HeatingQC', 'CentralAir',
                                                   'Electrical', ...])])),
                ('classifier', Lasso(alpha=100, max_iter=2000))]) """

# and find the best fit parameters like this
grid_search.best_params_
""" 
{'classifier__alpha': 100, 'preprocessor__categorical__imputer__strategy': 'constant', 
'preprocessor__numerical__imputer__strategy': 'median'} """

# here we can see all the combinations evaluated during the gridsearch
grid_search.cv_results_['params']
""" 
[{'classifier__alpha': 10, 'preprocessor__categorical__imputer__strategy': 'most_frequent', 
'preprocessor__numerical__imputer__strategy': 'mean'}, {'classifier__alpha': 10, 
'preprocessor__categorical__imputer__strategy': 'most_frequent', 'preprocessor__numerical__imputer__strategy': 'median'},
 {'classifier__alpha': 10, 'preprocessor__categorical__imputer__strategy': 'constant', 
 'preprocessor__numerical__imputer__strategy': 'mean'}, {'classifier__alpha': 10, 
 'preprocessor__categorical__imputer__strategy': 'constant', 'preprocessor__numerical__imputer__strategy': 'median'}, 
 {'classifier__alpha': 100, 'preprocessor__categorical__imputer__strategy': 'most_frequent', 
 'preprocessor__numerical__imputer__strategy': 'mean'}, {'classifier__alpha': 100, 
 'preprocessor__categorical__imputer__strategy': 'most_frequent', 'preprocessor__numerical__imputer__strategy': 'median'},
  {'classifier__alpha': 100, 'preprocessor__categorical__imputer__strategy': 'constant', 
  'preprocessor__numerical__imputer__strategy': 'mean'}, {'classifier__alpha': 100, 
  'preprocessor__categorical__imputer__strategy': 'constant', 'preprocessor__numerical__imputer__strategy': 'median'}, 
  {'classifier__alpha': 200, 'preprocessor__categorical__imputer__strategy': 'most_frequent', 
  'preprocessor__numerical__imputer__strategy': 'mean'}, {'classifier__alpha': 200, 
  'preprocessor__categorical__imputer__strategy': 'most_frequent', 
  'preprocessor__numerical__imputer__strategy': 'median'}, {'classifier__alpha': 200, 
  'preprocessor__categorical__imputer__strategy': 'constant', 'preprocessor__numerical__imputer__strategy': 'mean'}, 
  {'classifier__alpha': 200, 'preprocessor__categorical__imputer__strategy': 'constant',
   'preprocessor__numerical__imputer__strategy': 'median'}] """

# and here the scores for each of one of the above combinations
grid_search.cv_results_['mean_test_score']
""" array([0.84746254, 0.84739594, 0.84814964, 0.8481309 , 0.86624908,
       0.86621021, 0.86646886, 0.86651035, 0.86552764, 0.8654755 ,
       0.86525292, 0.86523714]) """

# and finally let's check the performance over the test set
print(("best linear regression from grid search: %.3f"% grid_search.score(X_test, y_test)))
""" 
best linear regression from grid search: 0.738

- This model overfits to the train set, look at the r2 of 0.93 obtained for the train set vs 0.738 for the test set.

- We will try to reduce this over-fitting in upcoming examples."""