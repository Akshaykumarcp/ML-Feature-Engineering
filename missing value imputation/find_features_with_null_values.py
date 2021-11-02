from matplotlib import colors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.dtypes.missing import isnull

# load dataset
data = pd.read_csv('J:\VSCODE_WORKSPACE\Feature-Engineering\dataset\\train.csv')

data.shape
# (1460, 81)

# find the features having missing data points or rows
features_with_na = [fea for fea in data.columns if data[fea].isnull().mean() > 0]
# ['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

# create dataframe with features having null values and percentage of null values

data_na = data[features_with_na].isnull().mean()
""" LotFrontage     0.177397
Alley           0.937671
MasVnrType      0.005479
MasVnrArea      0.005479
BsmtQual        0.025342
BsmtCond        0.025342
BsmtExposure    0.026027
BsmtFinType1    0.025342
BsmtFinType2    0.026027
Electrical      0.000685
FireplaceQu     0.472603
GarageType      0.055479
GarageYrBlt     0.055479
GarageFinish    0.055479
GarageQual      0.055479
GarageCond      0.055479
PoolQC          0.995205
Fence           0.807534
MiscFeature     0.963014 """

type(data_na)
# <class 'pandas.core.series.Series'>

# from series, lets create dataframe

data_na = pd.DataFrame(data_na.reset_index())
"""            index         0
0    LotFrontage  0.177397
1          Alley  0.937671
2     MasVnrType  0.005479
3     MasVnrArea  0.005479
4       BsmtQual  0.025342
5       BsmtCond  0.025342
6   BsmtExposure  0.026027
7   BsmtFinType1  0.025342
8   BsmtFinType2  0.026027
9     Electrical  0.000685
10   FireplaceQu  0.472603
11    GarageType  0.055479
12   GarageYrBlt  0.055479
13  GarageFinish  0.055479
14    GarageQual  0.055479
15    GarageCond  0.055479
16        PoolQC  0.995205
17         Fence  0.807534
18   MiscFeature  0.963014 """

type(data_na)
# <class 'pandas.core.frame.DataFrame'>

# there is no column names to the dataframe so, lets add column names

data_na.columns = ['feature','na_percentage']
""" feature  na_percentage
0    LotFrontage       0.177397
1          Alley       0.937671
2     MasVnrType       0.005479
3     MasVnrArea       0.005479
4       BsmtQual       0.025342
5       BsmtCond       0.025342
6   BsmtExposure       0.026027
7   BsmtFinType1       0.025342
8   BsmtFinType2       0.026027
9     Electrical       0.000685
10   FireplaceQu       0.472603
11    GarageType       0.055479
12   GarageYrBlt       0.055479
13  GarageFinish       0.055479
14    GarageQual       0.055479
15    GarageCond       0.055479
16        PoolQC       0.995205
17         Fence       0.807534
18   MiscFeature       0.963014 """