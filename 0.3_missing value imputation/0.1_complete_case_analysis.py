"""
COMPLETE CASE ANALYSIS [CAA]
- A.K.A list wise deletion of cases
- consists in discarding observations where values in any of the variables are missing
- complete case analysis means literally analysing only those observations for which there is information in all of the variables in the dataset.

CAN I APPLY CAA TO BOTH NUMERICAL AND CATEGORICAL FEATURE/VARIABLE ?
- yes... yes...

ASSUMPTIONS
- CCA works well when the data are missing completely at random (MCAR). 
- In fact, we should use CCA if we have reasons to believe that data is missing at random, and not otherwise. 
- When data is MCAR, excluding observations with missing information is in essence the same as randomly excluding some observations from the dataset. 
- Therefore the dataset after CCA is a fair representation of the original dataset.

ADVANTAGES
- Easy to implement
- No data manipulation required
- Preserves variable distribution (if data is MCAR, then the distribution of the variables of the reduced dataset should match the distribution in the original dataset)

DISADVANTAGES
- It can exclude a large fraction of the original dataset (if missing data is abundant)
- Excluded observations could be informative for the analysis (if data is not missing at random)
- CCA will create a biased dataset if the complete cases differ from the original data (e.g., when missing information is in fact MAR or NMAR and not missing at random).
- When using our models in production, the model will not know how to handle missing data

WHEN TO USE CCA ?
- Data is missing completely at random
- ~ 5% of the total dataset contains missing data

In many real life datasets, the amount of missing data is never small, and therefore CCA is typically never an option.

CCA AND MODELS IN PRODUCTION
- When using CCA, we remove all observations that contain missing information. 
- However, the data that we want to score with our model, may indeed contain missing information. 
- This will pose a problem when using our model in live systems, or as we call it, 
- when putting or models into production: when an observation contains missing data, the model will not be able to handle it.

- In order to avoid this problem, when putting models into production we need to do 1 of 2 things: 
- either we do not score observations with missing data, or we replace the missing values by another number. 
- We can choose any from the imputation techniques to replace NA in the data to be scored.

BELOW EXAMPLE ON HOW TO PERFORM CCA
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
"""
how to show all column names in VS CODE IDE ?

CREDITS: https://stackoverflow.com/questions/49188960/how-to-show-all-columns-names-on-a-large-pandas-dataframe

Method 1:

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

Method 2:

pd.options.display.max_columns = None
pd.options.display.max_rows = None
"""
# load dataset
data = pd.read_csv('dataset/house-prices-advanced-regression-techniques/train.csv')
data.shape
# (1460, 81)

# view dataset
data.head()
"""    Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \
0   1          60       RL         65.0     8450   Pave   NaN      Reg   1   2          20       RL         80.0     9600   Pave   NaN      Reg   
2   3          60       RL         68.0    11250   Pave   NaN      IR1   
3   4          70       RL         60.0     9550   Pave   NaN      IR1
4   5          60       RL         84.0    14260   Pave   NaN      IR1

  LandContour Utilities LotConfig LandSlope Neighborhood Condition1  \
0         Lvl    AllPub    Inside       Gtl      CollgCr       Norm
1         Lvl    AllPub       FR2       Gtl      Veenker      Feedr
2         Lvl    AllPub    Inside       Gtl      CollgCr       Norm
3         Lvl    AllPub    Corner       Gtl      Crawfor       Norm
4         Lvl    AllPub       FR2       Gtl      NoRidge       Norm

  Condition2 BldgType HouseStyle  OverallQual  OverallCond  YearBuilt  \
0       Norm     1Fam     2Story            7            5       2003
1       Norm     1Fam     1Story            6            8       1976
2       Norm     1Fam     2Story            7            5       2001
3       Norm     1Fam     2Story            7            5       1915
4       Norm     1Fam     2Story            8            5       2000

   YearRemodAdd RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType  \
0          2003     Gable  CompShg     VinylSd     VinylSd    BrkFace
1          1976     Gable  CompShg     MetalSd     MetalSd       None
2          2002     Gable  CompShg     VinylSd     VinylSd    BrkFace
3          1970     Gable  CompShg     Wd Sdng     Wd Shng       None
4          2000     Gable  CompShg     VinylSd     VinylSd    BrkFace

   MasVnrArea ExterQual ExterCond Foundation BsmtQual BsmtCond BsmtExposure  \
0       196.0        Gd        TA      PConc       Gd       TA           No
1         0.0        TA        TA     CBlock       Gd       TA           Gd
2       162.0        Gd        TA      PConc       Gd       TA           Mn
3         0.0        TA        TA     BrkTil       TA       Gd           No
4       350.0        Gd        TA      PConc       Gd       TA           Av

  BsmtFinType1  BsmtFinSF1 BsmtFinType2  BsmtFinSF2  BsmtUnfSF  TotalBsmtSF  \
0          GLQ         706          Unf           0        150          856
1          ALQ         978          Unf           0        284         1262
2          GLQ         486          Unf           0        434          920
3          ALQ         216          Unf           0        540          756
4          GLQ         655          Unf           0        490         1145

  Heating HeatingQC CentralAir Electrical  1stFlrSF  2ndFlrSF  LowQualFinSF  \
0    GasA        Ex          Y      SBrkr       856       854             0
1    GasA        Ex          Y      SBrkr      1262         0             0
2    GasA        Ex          Y      SBrkr       920       866             0
3    GasA        Gd          Y      SBrkr       961       756             0
4    GasA        Ex          Y      SBrkr      1145      1053             0

   GrLivArea  BsmtFullBath  BsmtHalfBath  FullBath  HalfBath  BedroomAbvGr  \
0       1710             1             0         2         1             3
1       1262             0             1         2         0             3
2       1786             1             0         2         1             3
3       1717             1             0         1         0             3
4       2198             1             0         2         1             4

   KitchenAbvGr KitchenQual  TotRmsAbvGrd Functional  Fireplaces FireplaceQu  \
0             1          Gd             8        Typ           0         NaN
1             1          TA             6        Typ           1          TA
2             1          Gd             6        Typ           1          TA
3             1          Gd             7        Typ           1          Gd
4             1          Gd             9        Typ           1          TA

  GarageType  GarageYrBlt GarageFinish  GarageCars  GarageArea GarageQual  \
0     Attchd       2003.0          RFn           2         548         TA
1     Attchd       1976.0          RFn           2         460         TA
2     Attchd       2001.0          RFn           2         608         TA
3     Detchd       1998.0          Unf           3         642         TA
4     Attchd       2000.0          RFn           3         836         TA

  GarageCond PavedDrive  WoodDeckSF  OpenPorchSF  EnclosedPorch  3SsnPorch  \
0         TA          Y           0           61              0          0
1         TA          Y         298            0              0          0
2         TA          Y           0           42              0          0
3         TA          Y           0           35            272          0
4         TA          Y         192           84              0          0

   ScreenPorch  PoolArea PoolQC Fence MiscFeature  MiscVal  MoSold  YrSold  \
0            0         0    NaN   NaN         NaN        0       2    2008
1            0         0    NaN   NaN         NaN        0       5    2007
2            0         0    NaN   NaN         NaN        0       9    2008
3            0         0    NaN   NaN         NaN        0       2    2006
4            0         0    NaN   NaN         NaN        0      12    2008

  SaleType SaleCondition  SalePrice
0       WD        Normal     208500
1       WD        Normal     181500
2       WD        Normal     223500
3       WD       Abnorml     140000
4       WD        Normal     250000 """

# FIND FEATURE/VARIABLE WITH MISSING OBERVATIONS/DATA POINTS

feat_with_na = [feat for feat in data.columns if data[feat].isnull().mean() > 0]
feat_with_na
# ['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

# FIND OUT THE DATA TYPE OF feat_with_na

data[feat_with_na].dtypes
""" LotFrontage     float64
Alley            object
MasVnrType       object
MasVnrArea      float64
BsmtQual         object
BsmtCond         object
BsmtExposure     object
BsmtFinType1     object
BsmtFinType2     object
Electrical       object
FireplaceQu      object
GarageType       object
GarageYrBlt     float64
GarageFinish     object
GarageQual       object
GarageCond       object
PoolQC           object
Fence            object
MiscFeature      object
dtype: object """

# both numerical & categorical data types are present. 

# view obervations/data points of missing data

data[feat_with_na].head(10)
""" LotFrontage Alley MasVnrType  MasVnrArea BsmtQual BsmtCond BsmtExposure  \
0         65.0   NaN    BrkFace       196.0       Gd       TA           No
1         80.0   NaN       None         0.0       Gd       TA           Gd
2         68.0   NaN    BrkFace       162.0       Gd       TA           Mn
3         60.0   NaN       None         0.0       TA       Gd           No
4         84.0   NaN    BrkFace       350.0       Gd       TA           Av
5         85.0   NaN       None         0.0       Gd       TA           No
6         75.0   NaN      Stone       186.0       Ex       TA           Av
7          NaN   NaN      Stone       240.0       Gd       TA           Mn
8         51.0   NaN       None         0.0       TA       TA           No
9         50.0   NaN       None         0.0       TA       TA           No

  BsmtFinType1 BsmtFinType2 Electrical FireplaceQu GarageType  GarageYrBlt  \
0          GLQ          Unf      SBrkr         NaN     Attchd       2003.0
1          ALQ          Unf      SBrkr          TA     Attchd       1976.0
2          GLQ          Unf      SBrkr          TA     Attchd       2001.0
3          ALQ          Unf      SBrkr          Gd     Detchd       1998.0
4          GLQ          Unf      SBrkr          TA     Attchd       2000.0
5          GLQ          Unf      SBrkr         NaN     Attchd       1993.0
6          GLQ          Unf      SBrkr          Gd     Attchd       2004.0
7          ALQ          BLQ      SBrkr          TA     Attchd       1973.0
8          Unf          Unf      FuseF          TA     Detchd       1931.0
9          GLQ          Unf      SBrkr          TA     Attchd       1939.0

  GarageFinish GarageQual GarageCond PoolQC  Fence MiscFeature
0          RFn         TA         TA    NaN    NaN         NaN
1          RFn         TA         TA    NaN    NaN         NaN
2          RFn         TA         TA    NaN    NaN         NaN
3          Unf         TA         TA    NaN    NaN         NaN
4          RFn         TA         TA    NaN    NaN         NaN
5          Unf         TA         TA    NaN  MnPrv        Shed
6          RFn         TA         TA    NaN    NaN         NaN
7          RFn         TA         TA    NaN    NaN        Shed
8          Unf         Fa         TA    NaN    NaN         NaN
9          RFn         Gd         TA    NaN    NaN         NaN """

# FIND OUT PERCENTAGE OF OBSERVATIONS/DATA POINTS PER FEATURE/VARIABLE

data_na_percentage = data[feat_with_na].isnull().mean()
""" LotFrontage     0.177397
Alley           0.937671
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
MiscFeature     0.963014
dtype: float64 """

# lets convert from float to dataframe 

# transform array into dataframe
data_na_percentage = pd.DataFrame(data_na_percentage.reset_index())
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

# add column names to dataframe
data_na_percentage.columns = ['feature','na_percentage']
"""          feature  na_percentage
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

# order based on descending
data_na_percentage.sort_values(by='na_percentage',ascending=False, inplace=True)
"""          feature  na_percentage
16        PoolQC       0.995205
18   MiscFeature       0.963014
1          Alley       0.937671
17         Fence       0.807534
10   FireplaceQu       0.472603
0    LotFrontage       0.177397
11    GarageType       0.055479
12   GarageYrBlt       0.055479
13  GarageFinish       0.055479
14    GarageQual       0.055479
15    GarageCond       0.055479
6   BsmtExposure       0.026027
8   BsmtFinType2       0.026027
7   BsmtFinType1       0.025342
5       BsmtCond       0.025342
4       BsmtQual       0.025342
3     MasVnrArea       0.005479
2     MasVnrType       0.005479
9     Electrical       0.000685 """

""" 
OBSERVATIONS:
- The first 6 variables contain a lot of missing information. 
- So we can't use CCA if we consider those variables, as most of the observations in the dataset will be discarded. 
- We could otherwise use CCA if we omit using those variables with a lot of NA.

- For this example, I will ignore the first 6 variables with a lot of missing data, and proceed with CCA in the remaining of the dataset. """

# lets capture features/variables with no or less than 5% NA to drop NA

features_cca = [feat for feat in data.columns if data[feat].isnull().mean() < 0.05]
""" ['Id', 'MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageCars', 'GarageArea', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice'] """

# PERCENTAGE OF OBSERVATIONS/DATA POINTS with complete cases i,e with values for all features/variables

(len(data[features_cca].dropna()) / len(data) *100)
# 96.7123287671233

data_cca = data[features_cca].dropna()
data.shape 
# (1460, 81)

data_cca.shape
# (1412, 70)

data_cca.hist(bins=50, density=True, figsize=(12,12))
plt.savefig("0._histogram_post_CCA.png")
plt.show()

# lets check distribution via histogram of few features/variables before & after CCA

fig = plt.figure()
ax = fig.add_subplot(111)

# original data
data['GrLivArea'].hist(bins=50, ax=ax, density = True, color='red')

# data after cca, the argument alpha makes the color transparent, so we can
# see the overlay of the 2 distributions
data_cca['GrLivArea'].hist(bins=50, ax=ax, color='blue', density=True, alpha=0.8)
plt.savefig('0._histogram_before_after_CCA_forFeature_GrLivArea.png')
plt.show()

# lets check distribution via density plot of few features/variables before & after CCA

fig = plt.figure()
ax = fig.add_subplot(111)

# original data
data['GrLivArea'].plot.density(color='red')

# data after cca
data_cca['GrLivArea'].plot.density(color='blue')
plt.savefig('0._densityPlot_before_after_CCA_forFeature_GrLivArea.png')
plt.show()

# let's check the distribution of a few variables before and after cca: histogram

fig = plt.figure()
ax = fig.add_subplot(111)

# original data
data['BsmtFinSF1'].hist(bins=50, ax=ax, density=True, color='red')

# data after cca, the argument alpha makes the color transparent, so we can
# see the overlay of the 2 distributions
data_cca['BsmtFinSF1'].hist(bins=50, ax=ax, color='blue', density=True, alpha=0.8)
plt.savefig('0._histogram_before_after_CCA_forFeature_BsmtFinSF1.png')
plt.show()

## let's check the distribution of a few variables before and after 
# cca: density plot

fig = plt.figure()
ax = fig.add_subplot(111)

# original data
data['BsmtFinSF1'].plot.density(color='red')

# data after cca
data_cca['BsmtFinSF1'].plot.density(color='blue')
plt.savefig('0._densityPlot_before_after_CCA_forFeature_BsmtFinSF1.png')
plt.show()

""" 
OBSERVATIONS:
- As we can see from the above plots, the distribution of the selected numerical variables in the original and complete case dataset is very similar
- which is what we expect from CCA if data is missing at random and only for a small proportion of the observations.

next up, explore the distribution of categorical variables. To do so, I will evaluate the percentage of observations that show each of the unique categories """

# the following function captures the percentage of observations for each category in the original and complete case dataset and puts them together in a new dataframe

def categorical_distribution(df, df_cca, variable):
  temp = pd.concat([
    # % of observations per category, original data
    df[variable].value_counts() / len(df),

    # % of observations per category, CCA data
    df_cca[variable].value_counts() / len(df_cca)
  ],axis=1)

  temp.column = ['original','cca']

  return temp

# run the function in a categorical variable
categorical_distribution(data, data_cca, 'BsmtQual')
"""     
    BsmtQual  BsmtQual
TA  0.444521  0.458924
Gd  0.423288  0.431303
Ex  0.082877  0.084986
Fa  0.023973  0.024788 """

categorical_distribution(data, data_cca, 'MasVnrType')
"""          MasVnrType  MasVnrType
None       0.591781    0.588527
BrkFace    0.304795    0.310198
Stone      0.087671    0.090652
BrkCmn     0.010274    0.010623 """

categorical_distribution(data, data_cca, 'SaleCondition')
"""          SaleCondition  SaleCondition
Normal        0.820548       0.820822
Partial       0.085616       0.086402
Abnorml       0.069178       0.070822
Family        0.013699       0.014164
Alloca        0.008219       0.005666
AdjLand       0.002740       0.002125 """

""" 
OBSERVATION
- As we can see from the output above, the distribution of houses in each of the categories, is very similar in the original and complete case dataset
- which again, is what is expected if the data is missing completely at random, and the percentage of missing data is small.
 """