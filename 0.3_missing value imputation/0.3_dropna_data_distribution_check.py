from matplotlib import colors
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pandas.core.dtypes.missing import isnull

# load dataset
data = pd.read_csv('dataset\\train.csv')

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

# lets address the NA data points or observations

# first, lets filter the NA's that are less than 5%

# select features with no or less than 5% NA

features_with_na_5perc = [feat for feat in data.columns if data[feat].isnull().mean() < 0.05]
""" ['Id', 'MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageCars', 'GarageArea', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice']
"""

# since only 5% of NA are present, lets drop them 

data2 = data[features_with_na_5perc].dropna()

# lets compare the difference after droping NA
data.shape
# (1460, 81)

data2.shape
# (1412, 70)

# due to dropping there may be difference in data distribution
# data distribution should be preserved even after data cleaning process 
# lets cross verify whether there is data distribution issue present or not

data2.hist(bins=50, density=True, figsize=(12,12))
plt.savefig('missing value imputation/0.3_dataset_histogram_after_removing_5perc_NA.png')
plt.show()

# lets compare distribution for few features before and after

# lets check out for 'GrLivArea' numerical feature

fig = plt.figure()

ax = fig.add_subplot(111)

# original data
data['GrLivArea'].hist(bins=50, ax=ax, density=True, color='red')
#plt.savefig("missing value imputation/data_distribution_before_after_for_GrLivArea.png")
#plt.show()

# preprocessed data
# alpha makes color transparent to see overlay of 2 distribution
data2['GrLivArea'].hist(bins=50, ax=ax, density=True, color='blue', alpha=0.8)
plt.savefig("missing value imputation/0.3_data_distribution_before_after_for_GrLivArea.png")
plt.show()

# by looking at the plot, data distribution is preserved

# lets have a look via density plot
data['GrLivArea'].plot.density(color='red')
plt.savefig("missing value imputation/0.3_original_data_GrLivArea_density_plot.png")
plt.show()

data2['GrLivArea'].plot.density(color='blue')
plt.savefig("missing value imputation/0.3_preprocessed_data_GrLivArea_density_plot.png")
plt.show()

# same sort of distribution comparison can be made for other numerical features as well

# how about categorical features ?
# for categorical features, we do frequency analysis

# lets check out below

# lets create a function that captures % of observations or data points
# for each category in original and preprocessed dataset
# and create a dataframe out of it

def categorical_distribution(original_data, preprocessed_data,feature):
    tmp = pd.concat(
        [
            # percentage of observations per category in original data
            original_data[feature].value_counts() / len(original_data),
            
            # percentage of observations per category in preprocessed data
            preprocessed_data[feature].value_counts() / len(preprocessed_data)
        ],            
        axis=1
    )

    # add column names for dataframe
    tmp.columns = ['original','preprocessed']

    return tmp

# call the function
categorical_distribution(data,data2,'BsmtQual')
"""     
    original  preprocessed
TA  0.444521      0.458924
Gd  0.423288      0.431303
Ex  0.082877      0.084986
Fa  0.023973      0.024788 """

# we shall see above, that disctribution for TA in original and preprocessed are roughly same 

# same sort of distribution comparison can be made for other categorical features as well