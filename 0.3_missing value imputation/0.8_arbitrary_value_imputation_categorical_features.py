""" 
Arbitrary value imputation for categorical variables
- This is the most widely used method of missing data imputation for categorical variables. 
- This method consists in treating missing data as an additional label or category of the variable. 
- All the missing observations are grouped in the newly created label 'Missing'.

- This is in essence, the equivalent of replacing by an arbitrary value for numerical variables.

- The beauty of this technique resides on the fact that it does not assume anything about the fact that the data is missing. 
- It is very well suited when the number of missing data is high.

Advantages
- Easy to implement
- Fast way of obtaining complete datasets
- Can be integrated in production (during model deployment)
- Captures the importance of "missingness" if there is one
- No assumption made on the data

Limitations
- If the number of NA is small, creating an additional category may cause trees to over-fit
- For categorical variables this is the method of choice, as it treats missing values as a separate category, without making any assumption on the variable or the reasons why data could be missing. 
- It is used widely in data science competitions and organisations. 
- See for example the winning solution of the KDD 2009 cup: "Winning the KDD Cup Orange Challenge with Ensemble Selection" (http://www.mtome.com/Publications/CiML/CiML-v3-book.pdf).

Below lets see example on how to perform frequent/mode category imputation on house price dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)

""" Arbitrary value imputation for categorical variables on House Prices dataset  """

# let's load the dataset with a few columns for the demonstration

# these are categorical columns and the target SalePrice
cols_to_use = ['BsmtQual', 'FireplaceQu', 'SalePrice']

data = pd.read_csv('dataset/house-prices-advanced-regression-techniques/train.csv', usecols=cols_to_use)
data.head()
""" 
  BsmtQual FireplaceQu  SalePrice
0       Gd         NaN     208500
1       Gd          TA     181500
2       Gd          TA     223500
3       TA          Gd     140000
4       Gd          TA     250000 """

# let's inspect the percentage of missing values in each variable
data.isnull().mean()
""" 
BsmtQual       0.025342
FireplaceQu    0.472603
SalePrice      0.000000
dtype: float64 

OBSERVATION:
- When replacing NA in categorical variables by a label called 'Missing', we are not learning anything from the training set, so in principle we could do this in the original dataset and then separate into train and test. 
- However, I do not recommend this practice. 
- You will see in upcoming examples that splitting into train and test right at the beginning helps with building a machine learning pipeline. 
- So I will continue with this practice here as well. """

# let's separate into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    data,
    data.SalePrice,
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape
# ((1022, 3), (438, 3))

# BsmtQual feature analysis

# let's remind ourselves of the % of missing values
X_train['BsmtQual'].isnull().mean()
# 0.023483365949119372

# let's inspect the number of observations per category in BsmtQual value_counts() counts the amount of houses that show each of the labels in the variable indicated below
X_train['BsmtQual'].value_counts().sort_values(ascending=False).plot.bar()
plt.xlabel('BsmtQual')
plt.ylabel('Number of houses')
plt.savefig("missing value imputation/0.8_barPlot_BsmtQualFeature_housePriceDataset.png")
plt.show()

# Let's fill na in both train and test
# I use the fillna() method from pandas
# with the argument inplace=True, I indicate to pandas
# that I want the values replaced in the same dataset
X_train['BsmtQual'].fillna('Missing', inplace=True)
X_test['BsmtQual'].fillna('Missing', inplace=True)

# let's plot the number of houses per category in the imputed variable
X_train['BsmtQual'].value_counts().sort_values(ascending=False).plot.bar()
plt.xlabel('BsmtQual')
plt.ylabel('Number of houses')
plt.savefig("missing value imputation/0.8_barPlot_BsmtQualFeature_afterImputation_housePriceDataset.png")
plt.show()

# let's plot the distribution of the target for the houses
# that show the different categories of the variable

fig = plt.figure()
ax = fig.add_subplot(111)

# a plot per category
X_train[X_train['BsmtQual']=='TA']['SalePrice'].plot(kind='kde', ax=ax)
X_train[X_train['BsmtQual']=='Gd']['SalePrice'].plot(kind='kde', ax=ax)
X_train[X_train['BsmtQual']=='Ex']['SalePrice'].plot(kind='kde', ax=ax)
X_train[X_train['BsmtQual']=='Missing']['SalePrice'].plot(kind='kde', ax=ax)
X_train[X_train['BsmtQual']=='Fa']['SalePrice'].plot(kind='kde', ax=ax)

# add the legend
lines, labels = ax.get_legend_handles_labels()
labels = ['TA', 'Gd', 'Ex', 'Missing', 'Fa']
ax.legend(lines, labels, loc='best')
plt.savefig("missing value imputation/0.8_distributionPlot_BsmtQualFeature_afterImputation_housePriceDataset.png")
plt.show()

"""
OBSERVATION: 
- You can see that the houses with different labels show different distributions of Prices. 
- For example the houses with the label Ex tend to be the most expensive, whereas the houses that show Missing or Fa are the cheapest. """

# FirePlaceQu feature analysis

# let's remind ourselves of the % of missing values
X_train['FireplaceQu'].isnull().mean()
# 0.46771037181996084

# let's inspect the number of observations per category
X_train['FireplaceQu'].value_counts().sort_values(ascending=False).plot.bar()
plt.xlabel('FireplaceQu')
plt.ylabel('Number of houses')
plt.savefig("missing value imputation/0.8_barPlot_FireplaceQuFeature_housePriceDataset.png")
plt.show()

# Let's fill na in both train and test
X_train['FireplaceQu'].fillna('Missing', inplace=True)
X_test['FireplaceQu'].fillna('Missing', inplace=True)

X_train['FireplaceQu'].value_counts().sort_values(ascending=False).plot.bar()
plt.xlabel('FireplaceQu')
plt.ylabel('Number of houses')
plt.savefig("missing value imputation/0.8_barPlot_FireplaceQuFeature_afterImputation_housePriceDataset.png")
plt.show()

""" 
OBSERVATION:
- We see now the additional category with the missing data: Missing
- This label contains most of the houses, as most showed missing data originally. """

# let's plot the distribution of the target for the houses
# that show the different categories of the variable

fig = plt.figure()
ax = fig.add_subplot(111)

# a plot per category
X_train[X_train['FireplaceQu']=='Missing']['SalePrice'].plot(kind='kde', ax=ax)
X_train[X_train['FireplaceQu']=='Gd']['SalePrice'].plot(kind='kde', ax=ax)
X_train[X_train['FireplaceQu']=='TA']['SalePrice'].plot(kind='kde', ax=ax)
X_train[X_train['FireplaceQu']=='Fa']['SalePrice'].plot(kind='kde', ax=ax)
X_train[X_train['FireplaceQu']=='Ex']['SalePrice'].plot(kind='kde', ax=ax)
X_train[X_train['FireplaceQu']=='Po']['SalePrice'].plot(kind='kde', ax=ax)


# add the legend
lines, labels = ax.get_legend_handles_labels()
labels = ['Missing','Gd', 'TA', 'Fa', 'Ex', 'Po']
ax.legend(lines, labels, loc='best')
plt.savefig("missing value imputation/0.8_distributionPlot_FireplaceQuFeature_afterImputation_housePriceDataset.png")
plt.show()

""" 
OBSERVATION:
- We observe again that the houses with different labels for FirePlaceQu also show different distributions of SalePrice, with those showing Ex being the most expensive ones and those showing Missing, or Fa being the cheapest ones. """

# here is a way of making it more general

def automate_plot(df, variable, target):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for category in df[variable].unique():
        df[df[variable]==category][target].plot(kind='kde', ax=ax)
    
    # add the legend
    lines, labels = ax.get_legend_handles_labels()
    labels = df[variable].unique()
    ax.legend(lines, labels, loc='best')
    plt.savefig("missing value imputation/0.8_automatePlot_distributionPlot_FireplaceQuFeature_afterImputation_housePriceDataset.png")    
    plt.show()

automate_plot(X_train, 'FireplaceQu', 'SalePrice')

automate_plot(X_train, 'BsmtQual', 'SalePrice')

# thats it for this example


# happy learning...