""" 
Gaussian Transformation with Scikit-learn
- Scikit-learn has recently released transformers to do Gaussian mappings as they call the variable transformations. 
        The PowerTransformer allows to do Box-Cox and Yeo-Johnson transformation. 
        With the FunctionTransformer, we can specify any function we want.

- The transformers per say, do not allow to select columns, but we can do so using a third transformer, 
        the ColumnTransformer

- Another thing to keep in mind is that Scikit-learn transformers return NumPy arrays, and not dataframes, 
        so we need to be mindful of the order of the columns not to mess up with our features.

Important
- Box-Cox and Yeo-Johnson transformations need to learn their parameters from the data. 
        Therefore, as always, before attempting any transformation it is important to divide the dataset into train 
        and test set.

- In this example, I will not do so for simplicity, but when using this transformation in your pipelines, 
        please make sure you do so.

In this example
We will see how to implement variable transformations using Scikit-learn and the House Prices dataset.
 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import FunctionTransformer, PowerTransformer

# load the data

data = pd.read_csv('dataset/house-prices-advanced-regression-techniques/train.csv')

data.head()
""" 
   Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape LandContour Utilities  ... PoolArea PoolQC Fence MiscFeature MiscVal MoSold YrSold  SaleType  SaleCondition  SalePrice
0   1          60       RL         65.0     8450   Pave   NaN      Reg         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   2008        WD         Normal     208500
1   2          20       RL         80.0     9600   Pave   NaN      Reg         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      5   2007        WD         Normal     181500
2   3          60       RL         68.0    11250   Pave   NaN      IR1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      9   2008        WD         Normal     223500
3   4          70       RL         60.0     9550   Pave   NaN      IR1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   2006        WD        Abnorml     140000
4   5          60       RL         84.0    14260   Pave   NaN      IR1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0     12   2008        WD         Normal     250000 """

""" 
- Let's select the numerical and positive variables in the dataset for this example. 
        As most of the transformations require the variables to be positive. """

cols = []

for col in data.columns:
    if data[col].dtypes != 'O' and col != 'Id':  # if the variable is numerical
        if np.sum(np.where(data[col] <= 0, 1, 0)) == 0:  # if the variable is positive
            cols.append(col)  # append variable to the list

cols
""" ['MSSubClass',
 'LotFrontage',
 'LotArea',
 'OverallQual',
 'OverallCond',
 'YearBuilt',
 'YearRemodAdd',
 '1stFlrSF',
 'GrLivArea',
 'TotRmsAbvGrd',
 'GarageYrBlt',
 'MoSold',
 'YrSold',
 'SalePrice'] """

# let's explore the distribution of the numerical variables
data[cols].hist(figsize=(20,20))
plt.show()

""" 
Plots to assess normality
- To visualise the distribution of the variables, we plot a histogram and a Q-Q plot. 
        In the Q-Q pLots, if the variable is normally distributed, the values of the variable should fall in a 
        45 degree line when plotted against the theoretical quantiles. 
 """

# plot the histograms to have a quick look at the variable distribution
# histogram and Q-Q plots

def diagnostic_plots(df, variable):
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist(bins=30)
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.show()

# Logarithmic transformation
# create a log transformer

transformer = FunctionTransformer(np.log, validate=True)
# transform all the numerical and positive variables

data_t = transformer.transform(data[cols].fillna(1))
# Scikit-learn returns NumPy arrays, so capture in dataframe
# note that Scikit-learn will return an array with
# only the columns indicated in cols

data_t = pd.DataFrame(data_t, columns = cols)

# original distribution
diagnostic_plots(data, 'GrLivArea')

# transformed distribution
diagnostic_plots(data_t, 'GrLivArea')

# original distribution
diagnostic_plots(data, 'MSSubClass')

# transformed distribution
diagnostic_plots(data_t, 'MSSubClass')

# Reciprocal transformation
# create the transformer
transformer = FunctionTransformer(lambda x: 1/x, validate=True)

# also # transformer = FunctionTransformer(np.reciprocal, validate=True)

# transform the positive variables
data_t = transformer.transform(data[cols].fillna(1))

# re-capture in a dataframe
data_t = pd.DataFrame(data_t, columns = cols)

# transformed variable
diagnostic_plots(data_t, 'GrLivArea')

# transformed variable
diagnostic_plots(data_t, 'MSSubClass')

# Square root transformation
transformer = FunctionTransformer(lambda x: x**(1/2), validate=True)

# also
# transformer = FunctionTransformer(np.sqrt, validate=True)

data_t = transformer.transform(data[cols].fillna(1))

data_t = pd.DataFrame(data_t, columns = cols)
diagnostic_plots(data_t, 'GrLivArea')

diagnostic_plots(data_t, 'MSSubClass')

# Exponential

transformer = FunctionTransformer(lambda x: x**(1/1.2), validate=True)

data_t = transformer.transform(data[cols].fillna(1))

data_t = pd.DataFrame(data_t, columns = cols)
diagnostic_plots(data_t, 'GrLivArea')

diagnostic_plots(data_t, 'MSSubClass')

# Box-Cox transformation
# create the transformer
transformer = PowerTransformer(method='box-cox', standardize=False)

# find the optimal lambda using the train set
transformer.fit(data[cols].fillna(1))

# transform the data
data_t = transformer.transform(data[cols].fillna(1))

# capture data in a dataframe
data_t = pd.DataFrame(data_t, columns = cols)

diagnostic_plots(data_t, 'GrLivArea')

diagnostic_plots(data_t, 'MSSubClass')

""" 
Yeo-Johnson
- Yeo-Johnson is an adaptation of Box-Cox that can also be used in negative value variables. 
        So let's expand the list of variables for the example, to include those that contain zero and negative 
        values as well.
 """
cols = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
    'OverallCond', 'MasVnrArea', 'BsmtFinSF1',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
    'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
    'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
    'MiscVal', 'SalePrice'
]
# call the transformer
transformer = PowerTransformer(method='yeo-johnson', standardize=False)

# learn the lambda from the train set
transformer.fit(data[cols].fillna(1))

# transform the data
data_t = transformer.transform(data[cols].fillna(1))

# capture data in a dataframe
data_t = pd.DataFrame(data_t, columns = cols)
diagnostic_plots(data_t, 'GrLivArea')

diagnostic_plots(data_t, 'MSSubClass')
