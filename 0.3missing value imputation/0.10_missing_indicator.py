""" 
MISSING INDICATOR
- Adding a variable to capture NA
- In previous examples we learnt how to replace missing values by the mean, median or by extracting a random value. 
- In other words we learnt about mean / median and random sample imputation. 
- These methods assume that the data are missing completely at random (MCAR).

- There are other methods that can be used when values are not missing at random, 
- for example arbitrary value imputation or end of distribution imputation. 
- However, these imputation techniques will affect the variable distribution dramatically, 
- and are therefore not suitable for linear models.

So what can we do if data are not MCAR and we want to use linear models?
- If data are not missing at random, it is a good idea to replace missing observations by the mean / median / mode AND flag those missing observations as well with a Missing Indicator. 
- A Missing Indicator is an additional binary variable, which indicates whether the data was missing for an observation (1) or not (0).

For which variables can I add a missing indicator?
- We can add a missing indicator to both numerical and categorical variables.

Note
- Adding a missing indicator is never used alone. 
- On the contrary, it is always used together with another imputation technique, which can be mean / median imputation for numerical variables, or frequent category imputation for categorical variables. 
- We can also use random sample imputation together with adding a missing indicator for both categorical and numerical variables.

Commonly used together:
- Mean / median imputation + missing indicator (Numerical variables)
- Frequent category imputation + missing indicator (Categorical variables)
- Random sample Imputation + missing indicator (Numerical and categorical)

Assumptions
- Data is not missing at random
- Missing data are predictive

Advantages
- Easy to implement
- Captures the importance of missing data if there is one

Limitations
- Expands the feature space
- Original variable still needs to be imputed to remove the NaN
- Adding a missing indicator will increase 1 variable per variable in the dataset with missing values. So if the dataset contains 10 features, and all of them have missing values, after adding a missing indicator we will have a dataset with 20 features: the original 10 features plus additional 10 binary features, which indicate for each of the original variables whether the value was missing or not. This may not be a problem in datasets with tens to a few hundreds variables, but if our original dataset contains thousands of variables, by creating an additional variable to indicate NA, we will end up with very big datasets.

Important
- In addition, data tends to be missing for the same observation across multiple variables, which often leads to many of the missing indicator variables to be actually similar or identical to each other.

Final note
- Typically, mean / median / mode imputation is done together with adding a variable to capture those observations where the data was missing, thus covering 2 angles: if the data was missing completely at random, this would be contemplated by the mean / median / mode imputation, and if it wasn't this would be captured by the missing indicator.

- Both methods are extremely straight forward to implement, and therefore are a top choice in data science competitions. See for example the winning solution of the KDD 2009 cup: "Winning the KDD Cup Orange Challenge with Ensemble Selection. 

Below lets see example on how to perform missing indicator imputation on house price and titanic dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)

""" Missing indicator on titanic dataset """

# load the Titanic Dataset with a few variables for demonstration
data = pd.read_csv('dataset/titanic.csv', usecols=['age', 'fare', 'survived'])

data.head()

""" 
   survived      age      fare
0         1  29.0000  211.3375
1         1   0.9167  151.5500
2         0   2.0000  151.5500
3         0  30.0000  151.5500
4         0  25.0000  151.5500 """

# let's look at the percentage of NA
data.isnull().mean()
""" 
survived    0.000000
age         0.200917
fare        0.000764
dtype: float64 """

""" 
- To add a binary missing indicator, we don't necessarily need to learn anything from the training set, so in principle we could do this in the original dataset and then separate into train and test. 
- However, I do not recommend this practice. 
- In addition, if you are using scikit-learn to add the missing indicator, the indicator as it is designed, needs to learn from the train set, which features to impute, this is, which are the features for which the binary variable needs to be added. We will see more about different implementations of missing indicators in future examples. 
- For now, let's see how to create a binary missing indicator manually. """

# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(
    data[['age', 'fare']],  # predictors
    data['survived'],  # target
    test_size=0.3,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape
# ((916, 2), (393, 2))

# Let's explore the missing data in the train set the percentages should be fairly similar to those of the whole dataset
X_train.isnull().mean()
""" age     0.191048
fare    0.000000
dtype: float64 """

# add the missing indicator

# this is done very simply by using np.where from numpy and isnull from pandas:
X_train['Age_NA'] = np.where(X_train['age'].isnull(), 1, 0)
X_test['Age_NA'] = np.where(X_test['age'].isnull(), 1, 0)

X_train.head()
""" 
       age     fare  Age_NA
501   13.0  19.5000       0
588    4.0  23.0000       0
402   30.0  13.8583       0
1193   NaN   7.7250       1
686   22.0   7.7250       0 """

# the mean of the binary variable, coincides with the 
# perentage of missing values in the original variable
X_train['Age_NA'].mean()
# 0.19104803493449782

# yet the original variable, still shows the missing values which need to be replaced by any of the techniques we have learnt
X_train.isnull().mean()
""" 
age       0.191048
fare      0.000000
Age_NA    0.000000
dtype: float64 """

# for example median imputation

median = X_train['age'].median()

X_train['age'] = X_train['age'].fillna(median)
X_test['age'] = X_test['age'].fillna(median)

# check that there are no more missing values
X_train.isnull().mean()
""" 
age       0.0
fare      0.0
Age_NA    0.0
dtype: float64 """

""" Missing indicator on House Prices dataset """

# we are going to use the following variables, some are categorical some are numerical

cols_to_use = [
    'LotFrontage', 'MasVnrArea', # numerical
    'BsmtQual', 'FireplaceQu', # categorical
    'SalePrice' # target
]

data = pd.read_csv('dataset/house-prices-advanced-regression-techniques/train.csv', usecols=cols_to_use)
data.head()
""" 
   LotFrontage  MasVnrArea BsmtQual FireplaceQu  SalePrice
0         65.0       196.0       Gd         NaN     208500
1         80.0         0.0       Gd          TA     181500
2         68.0       162.0       Gd          TA     223500
3         60.0         0.0       TA          Gd     140000
4         84.0       350.0       Gd          TA     250000 """

# let's inspect the variables with missing values
data.isnull().mean()
""" 
LotFrontage    0.177397
MasVnrArea     0.005479
BsmtQual       0.025342
FireplaceQu    0.472603
SalePrice      0.000000
dtype: float64 """

# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(data,
                                                    data['SalePrice'],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
# ((1022, 5), (438, 5))

# let's make a function to add a missing indicator binary variable

def missing_indicator(df, variable):    
    return np.where(df[variable].isnull(), 1, 0)

# let's loop over all the variables and add a binary 
# missing indicator with the function we created

for variable in cols_to_use:
    X_train[variable+'_NA'] = missing_indicator(X_train, variable)
    X_test[variable+'_NA'] = missing_indicator(X_test, variable)
    
X_train.head()
""" 
      LotFrontage  MasVnrArea BsmtQual FireplaceQu  SalePrice  LotFrontage_NA  \
64            NaN       573.0       Gd         NaN     219500               1
682           NaN         0.0       Gd          Gd     173000               1
960          50.0         0.0       TA         NaN     116500               0
1384         60.0         0.0       TA         NaN     105000               0
1100         60.0         0.0       TA         NaN      60000               0

      MasVnrArea_NA  BsmtQual_NA  FireplaceQu_NA  SalePrice_NA
64                0            0               1             0
682               0            0               0             0
960               0            0               1             0
1384              0            0               1             0
1100              0            0               1             0 """

# now let's evaluate the mean value of the missing indicators

# first I capture the missing indicator variables with a 
# list comprehension
missing_ind = [col for col in X_train.columns if 'NA' in col]

# calculate the mean
X_train[missing_ind].mean()
""" 
LotFrontage_NA    0.184932
MasVnrArea_NA     0.004892
BsmtQual_NA       0.023483
FireplaceQu_NA    0.467710
SalePrice_NA      0.000000
dtype: float64 """

# the mean of the missing indicator coincides with the percentage of missing values in the original variable
X_train.isnull().mean()
""" 
LotFrontage       0.184932
MasVnrArea        0.004892
BsmtQual          0.023483
FireplaceQu       0.467710
SalePrice         0.000000
LotFrontage_NA    0.000000
MasVnrArea_NA     0.000000
BsmtQual_NA       0.000000
FireplaceQu_NA    0.000000
SalePrice_NA      0.000000
dtype: float64 """

# let's make a function to fill missing values with a value:  we have use a similar function in our previous examples so you are probably familiar with it

def impute_na(df, variable, value):
    return df[variable].fillna(value)

# let's impute the NA with  the median for numerical variables remember that we calculate the median using the train set

median = X_train['LotFrontage'].median()
X_train['LotFrontage'] = impute_na(X_train, 'LotFrontage', median)
X_test['LotFrontage'] = impute_na(X_test, 'LotFrontage', median)

median = X_train['MasVnrArea'].median()
X_train['MasVnrArea'] = impute_na(X_train, 'MasVnrArea', median)
X_test['MasVnrArea'] = impute_na(X_test, 'MasVnrArea', median)

# let's impute the NA in categorical variables by the most frequent category (aka the mode) the mode needs to be learnt from the train set

mode = X_train['BsmtQual'].mode()[0]
X_train['BsmtQual'] = impute_na(X_train, 'BsmtQual', mode)
X_test['BsmtQual'] = impute_na(X_test, 'BsmtQual', mode)

mode = X_train['FireplaceQu'].mode()[0]
X_train['FireplaceQu'] = impute_na(X_train, 'FireplaceQu', mode)
X_test['FireplaceQu'] = impute_na(X_test, 'FireplaceQu', mode)

# and now let's check there are no more NA
X_train.isnull().mean()
""" LotFrontage       0.0
MasVnrArea        0.0
BsmtQual          0.0
FireplaceQu       0.0
SalePrice         0.0
LotFrontage_NA    0.0
MasVnrArea_NA     0.0
BsmtQual_NA       0.0
FireplaceQu_NA    0.0
SalePrice_NA      0.0
dtype: float64 """

""" 
OBSERVATION:
As you can see, we have now the double of features respect to the original dataset. The original dataset had 4 variables, the pre-processed dataset contains 8, plus the target. 

Thats it for now, happy learning"""