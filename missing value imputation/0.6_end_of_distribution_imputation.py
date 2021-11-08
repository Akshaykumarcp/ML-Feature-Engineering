"""
End of distribution imputation
- In the previous python example we replaced missing data by an arbitrary value. 
- However, determining the value of the arbitrary value can be laborious and it is usually a manual job. 
- We can automate this process by automatically selecting arbitrary values at the end of the variable distributions.

How do we select the value at the end?
- If the variable is normally distributed, we can use the mean plus or minus 3 times the standard deviation
- If the variable is skewed, we can use the IQR proximity rule
- We can also select the min / max value and multiply it by a certain amount of times, like 2 or 3.

Which variables can I impute with an arbitrary value?
- This method is suitable for numerical variables.

Assumptions
- MNAR - values are not missing at random
- If the value is not missing at random, we don't want to replace it for the mean / median and therefore make that observation look like the majority of our observations. 
- Instead, we want to flag that observation as different, and therefore we assign a value that is at the tail of the distribution, where observations are rarely represented in the population.

Advantages
- Easy to implement
- Fast way of obtaining complete datasets
- Can be integrated in production (during model deployment)
- Captures the importance of "missingess" if there is one

Disadvantages
- Distortion of the original variable distribution
- Distortion of the original variance
- Distortion of the covariance with the remaining variables of the dataset
- This technique may mask true outliers in the distribution

Final note
- I haven't seen this method used in data competitions, 
- however, this method is used in finance companies. 
- When capturing the financial history of customers, in order not to assume that missing is at random, the missing data are replaced by a value at the end of the distribution.

Let's see example below on how to perform end of distribution imputation on Ames House Price and Titanic Datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)

""" End of distribution imputation on Titanic dataset """

# load house price dataset
data = pd.read_csv('dataset/titanic.csv', usecols=['age', 'fare', 'survived'])
data.head()
""" 
   survived      age      fare
0         1  29.0000  211.3375
1         1   0.9167  151.55002         0   2.0000  151.5500
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
Imputation important
- Imputation has to be done over the training set, and then propagated to the test set. 
- For this imputation technique, this means that when selecting the value with which we will replace the NA, we need to do so only by looking at the distribution of the variables in the training set. 
- Then we use the selected value to replace NA both in the train and test set. """

# let's separate into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    data[['age', 'fare']],  # predictors
    data['survived'],  # target
    test_size=0.3,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape
# ((916, 2), (393, 2))

# let's have a look at the distribution of Ages
X_train.age.hist(bins=50)
plt.savefig('missing value imputation/0.6_histogram_titanic_xtrain.png')
plt.show()

# let's make a function to fill missing values with an extreme value: the variable takes the dataframe, the variable, and the value to replace na and returns the variable with the filled na
def impute_na(df, variable, value):
    return df[variable].fillna(value)

# far end of the distribution:
# Because Age looks approximately Gaussian, I use the
# mean and std to calculate the replacement value
X_train.age.mean() + 3 * X_train.age.std()
# 72.03416424092518

# replace NA with the value calculated above
X_train['Age_imputed'] = impute_na(X_train, 'age',
                                   X_train.age.mean() + 3 * X_train.age.std())

X_train.head(15)
""" 
       age      fare  Age_imputed
501   13.0   19.5000    13.000000
588    4.0   23.0000     4.000000
402   30.0   13.8583    30.000000
1193   NaN    7.7250    72.034164
686   22.0    7.7250    22.000000
971    NaN    7.8792    72.034164
117   30.0   56.9292    30.000000
540    2.0   26.0000     2.000000
294   49.0  110.8833    49.000000
261   35.0   26.2875    35.000000
587    2.0   23.0000     2.000000
489   42.0   26.0000    42.000000
2      2.0  151.5500     2.000000
405   18.0   13.0000    18.000000
1284   NaN    8.0500    72.034164 """

# we can see a change in the variance after end of tail imputation
# this is expected, because the percentage of missing data is quite
# high in Age ~20%
print('Original variable variance: ', X_train['age'].var())
print('Variance after imputation: ', X_train['Age_imputed'].var())
""" Original variable variance:  194.16304666581863
Variance after imputation:  427.39198372523526 """

# we can see that the distribution has changed 
# with now more values accumulating towards the tail

fig = plt.figure()
ax = fig.add_subplot(111)

# original variable distribution
X_train['age'].plot(kind='kde', ax=ax)

# imputed variable
X_train['Age_imputed'].plot(kind='kde', ax=ax, color='red')

# add legends
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.savefig('missing value imputation/0.6_titanic_ageFeatureBeforeAndAfter_distribution_imputation.png')
plt.show()
""" 
OBSERVATION:
- As mentioned above, similarly to arbitrary value imputation, end of tail imputation distorts the original distribution of the variable Age. 
- The transformed variable shows more values around selected tail value. 

Is this important?
- It depends on the machine learning model you want to build. 
- Linear models assume that the variables are normally distributed. 
- End of tail imputation may distort the original normal distribution if the % of missing data is high. 
- Therefore the final imputed variable will no longer be normally distributed, which in turn may affect the linear model performance.

- On the other hand, this technique works quite well with tree based algorithms."""

# we also said end of tail imputation may affect the relationship 
# with the other variables in the dataset, let's have a look

X_train[['fare', 'age', 'Age_imputed']].cov()
""" 
                    fare         age  Age_imputed
fare         2248.326729  136.176223    19.647139
age           136.176223  194.163047   194.163047
Age_imputed    19.647139  194.163047   427.391984

OBSERVATION:
We see indeed that the covariance between Age and Fare is changed after the arbitrary value imputation. """

# Finally, I mentioned that end tail imputation may affect the perception of outliers

# Let's find out using a boxplot
X_train[['age', 'Age_imputed']].boxplot()
plt.savefig('missing value imputation/0.6_boxplot_titanic_ageFeatureBeforeAndAfter_imputation.png')
plt.show()
""" 
OBSERVATION:
Masks the outliers!! """

""" End of distribution imputation on Ames House Price """

# we are going to use only the following variables,
# 3 of which contain NA
cols_to_use = [
    'OverallQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'WoodDeckSF',
    'BsmtUnfSF', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'SalePrice'
]

# load data
data = pd.read_csv('dataset/house-prices-advanced-regression-techniques/train.csv', usecols=cols_to_use)

data.head()
""" 
   LotFrontage  OverallQual  MasVnrArea  BsmtUnfSF  TotalBsmtSF  1stFlrSF  \
0         65.0            7       196.0        150          856       856
1         80.0            6         0.0        284         1262      1262
2         68.0            7       162.0        434          920       920
3         60.0            7         0.0        540          756       961
4         84.0            8       350.0        490         1145      1145

   GrLivArea  GarageYrBlt  WoodDeckSF  SalePrice
0       1710       2003.0           0     208500
1       1262       1976.0         298     181500
2       1786       2001.0           0     223500
3       1717       1998.0           0     140000
4       2198       2000.0         192     250000 """

data.shape
# (1460, 10)

[feat for feat in data.columns if data[feat].isnull().sum() > 0]
# ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']

# find the percentage of missing data within those variables
data[['LotFrontage', 'MasVnrArea', 'GarageYrBlt']].isnull().mean()
""" LotFrontage    0.177397
MasVnrArea     0.005479
GarageYrBlt    0.055479
dtype: float64 """

# let's separate into training and testing set
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    data['SalePrice'],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
# ((1022, 10), (438, 10))

# let's plot the distributions of the variables
# we learnt this code in section 3 on variable characteristics

X_train[['LotFrontage', 'MasVnrArea', 'GarageYrBlt']].hist(bins=50, figsize=(10,10))
plt.savefig('missing value imputation/0.6_histogram_housePriceDataset.png')
plt.show()

""" the variables are not normally distributed
let's impute the NA using the IQR proximity rule as learnt from above section"""

# calulate the IQR
IQR = X_train['LotFrontage'].quantile(0.75) - X_train['LotFrontage'].quantile(
    0.25)
IQR
# 22.0

# calculate the upper boundary
extreme_value = X_train['LotFrontage'].quantile(0.75) + 3 * IQR
extreme_value
# 146.0

# let's impute the NA with the extreme value

X_train.loc[:,'LotFrontage_imputed'] = impute_na(X_train, 'LotFrontage', extreme_value)
X_test.loc[:,'LotFrontage_imputed'] = impute_na(X_test, 'LotFrontage', extreme_value)

# let's do the same for MasVnrArea

# calculate the IQR
IQR = X_train['MasVnrArea'].quantile(0.75) - X_train['MasVnrArea'].quantile(
    0.25)
IQR
# 170.0

# calculate the upper boundary
extreme_value = X_train['MasVnrArea'].quantile(0.75) + 3 * IQR
extreme_value
# 680.0

# let's impute the NA with the extreme value

X_train.loc[:,'MasVnrArea_imputed'] = impute_na(X_train, 'MasVnrArea', extreme_value)
X_test.loc[:,'MasVnrArea_imputed'] = impute_na(X_test, 'MasVnrArea', extreme_value)

# let's evaluate the effect of end tail imputation on the distribution

# we can see that the distribution has changed for LotFrontAge
# with now more values accumulating towards the extreme value

fig = plt.figure()
ax = fig.add_subplot(111)

# original variable distribution
X_train['LotFrontage'].plot(kind='kde', ax=ax)

# imputed variable
X_train['LotFrontage_imputed'].plot(kind='kde', ax=ax, color='red')

# add legends
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.savefig('missing value imputation/0.6_distribution_housePriceDataset_LotFrontageFeature_beforeAndAfterImputation.png')
plt.show()

# let's evaluate the effect of arbitrary imputation on the distribution

# MasVnrArea had only few outliers, so the change in the distribution is
# not so dramatic. Less than when using an arbitrary value of 2999 as
# we did in the previous notebook

fig = plt.figure()
ax = fig.add_subplot(111)

# original variable distribution
X_train['MasVnrArea'].plot(kind='kde', ax=ax)

# imputed variable
X_train['MasVnrArea_imputed'].plot(kind='kde', ax=ax, color='red')

# add legends
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.savefig('missing value imputation/0.6_distribution_housePriceDataset_MasVnrAreaFeature_beforeAndAfterImputation.png')
plt.show()

""" 
OBSERVATION:
- From the above plots we can see that the distribution of LotFrontAge is changed quite dramatically, but not so much the distribution of MasVnrArea. 
- This is because the % of missing values in the second variable is quite small. """

# we see that there are a fewer outliers as well after the imputation
X_train[['LotFrontage', 'LotFrontage_imputed']].boxplot()
plt.savefig('missing value imputation/0.6_boxplot_housePriceDataset_LotFrontageFeature_beforeAndAfterImputation.png')
plt.show()

# however, the outliers are not so affected for MasVnrArea
X_train[['MasVnrArea', 'MasVnrArea_imputed']].boxplot()
plt.savefig('missing value imputation/0.6_boxplot_housePriceDataset_MasVnrAreaFeature_beforeAndAfterImputation.png')
plt.show()

# similarly we can explore the effect of the imputation technique on 
# the variance

# we can see a change in the variance after imputation for LotFrontAge
# this is expected, because the percentage of missing data is quite
# high ~20%
print('Original Variance: ', X_train['LotFrontage'].var())
print('Variance after imputation: ', X_train['LotFrontage_imputed'].var())
""" Original Variance:  532.5872021885677
Variance after imputation:  1313.0936747097642 """

# the same for MasnVnrArea is not so big
# Note particularly, that this effect is smaller than the one we observed
# when imputing by 2999 in the previous notebook!!!
print('Original Variance: ', X_train['MasVnrArea'].var())
print('Variance after imputation: ', X_train['MasVnrArea_imputed'].var())
""" Original Variance:  32983.53871003956
Variance after imputation:  34441.331260745486
 """

# finally, let's explore the covariance:
# take your time to compare the values in the table below.
X_train.cov()
""" 
                       LotFrontage   OverallQual    MasVnrArea     BsmtUnfSF  \
LotFrontage             532.587202      6.587119  6.805603e+02  9.496573e+02
OverallQual               6.587119      1.843859  1.014970e+02  1.746147e+02
MasVnrArea              680.560330    101.496976  3.298354e+04  7.540788e+03
BsmtUnfSF               949.657293    174.614725  7.540788e+03  1.875241e+05
TotalBsmtSF            2908.855504    288.624075  2.478877e+04  7.513307e+04
1stFlrSF               3379.793504    224.297266  2.086595e+04  4.987449e+04
GrLivArea              3919.951834    409.124216  3.520785e+04  5.203392e+04
GarageYrBlt              30.611717     17.902809  1.203584e+03  1.823065e+03
WoodDeckSF              134.741376     31.685571  3.208924e+03 -1.833201e+03
SalePrice            668964.454191  83201.317781  6.836439e+06  6.833028e+06
LotFrontage_imputed     532.587202      3.425501  6.391007e+02 -8.507392e+02
MasVnrArea_imputed      693.487235    103.599142  3.298354e+04  7.680598e+03

                      TotalBsmtSF      1stFlrSF     GrLivArea    GarageYrBlt  \
LotFrontage          2.908856e+03  3.379794e+03  3.919952e+03      30.611717
OverallQual          2.886241e+02  2.242973e+02  4.091242e+02      17.902809
MasVnrArea           2.478877e+04  2.086595e+04  3.520785e+04    1203.583792
BsmtUnfSF            7.513307e+04  4.987449e+04  5.203392e+04    1823.065167
TotalBsmtSF          1.682931e+05  1.212079e+05  8.615192e+04    3173.042442
1stFlrSF             1.212079e+05  1.398656e+05  1.044401e+05    2009.195552
GrLivArea            8.615192e+04  1.044401e+05  2.681277e+05    2738.982988
GarageYrBlt          3.173042e+03  2.009196e+03  2.738983e+03     624.305948
WoodDeckSF           1.227966e+04  1.109406e+04  1.558395e+04     665.891118
SalePrice            2.003928e+07  1.783631e+07  2.934477e+07  930935.489321
LotFrontage_imputed  2.308793e+03  3.142725e+03  3.723250e+03      26.484608
MasVnrArea_imputed   2.473980e+04  2.055453e+04  3.541558e+04    1274.749707

                       WoodDeckSF     SalePrice  LotFrontage_imputed  \
LotFrontage          1.347414e+02  6.689645e+05           532.587202
OverallQual          3.168557e+01  8.320132e+04             3.425501
MasVnrArea           3.208924e+03  6.836439e+06           639.100679
BsmtUnfSF           -1.833201e+03  6.833028e+06          -850.739199
TotalBsmtSF          1.227966e+04  2.003928e+07          2308.792907
1stFlrSF             1.109406e+04  1.783631e+07          3142.724647
GrLivArea            1.558395e+04  2.934477e+07          3723.250100
GarageYrBlt          6.658911e+02  9.309355e+05            26.484608
WoodDeckSF           1.648582e+04  3.029981e+06           542.437312
SalePrice            3.029981e+06  6.105731e+09        578446.035624
LotFrontage_imputed  5.424373e+02  5.784460e+05          1313.093675
MasVnrArea_imputed   3.088048e+03  6.872641e+06           652.387102

                     MasVnrArea_imputed
LotFrontage                6.934872e+02
OverallQual                1.035991e+02
MasVnrArea                 3.298354e+04
BsmtUnfSF                  7.680598e+03
TotalBsmtSF                2.473980e+04
1stFlrSF                   2.055453e+04
GrLivArea                  3.541558e+04
GarageYrBlt                1.274750e+03
WoodDeckSF                 3.088048e+03
SalePrice                  6.872641e+06
LotFrontage_imputed        6.523871e+02
MasVnrArea_imputed         3.444133e+04 """

# thats it for this example.
# happy learning