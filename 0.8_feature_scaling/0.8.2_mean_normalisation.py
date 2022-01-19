""" 
Feature Scaling
- We discussed previously that the scale of the features is an important consideration when building machine 
        learning models. 

Briefly:

Feature magnitude matters because:
- The regression coefficients of linear models are directly influenced by the scale of the variable.
- Variables with bigger magnitude / larger value range dominate over those with smaller magnitude / value range
- Gradient descent converges faster when features are on similar scales
- Feature scaling helps decrease the time to find support vectors for SVMs
- Euclidean distances are sensitive to feature magnitude.
- Some algorithms, like PCA require the features to be centered at 0.

The machine learning models affected by the feature scale are:
- Linear and Logistic Regression
- Neural Networks
- Support Vector Machines
- KNN
- K-means clustering
- Linear Discriminant Analysis (LDA)
- Principal Component Analysis (PCA)

Feature Scaling
- Feature scaling refers to the methods or techniques used to normalize the range of independent variables in our 
        data, or in other words, the methods to set the feature value range within a similar scale. 
- Feature scaling is generally the last step in the data preprocessing pipeline, performed just before training the 
        machine learning algorithms.

There are several Feature Scaling techniques, which we will discuss throughout this section:
- Standardisation
- Mean normalisation
- Scaling to minimum and maximum values - MinMaxScaling
- Scaling to maximum value - MaxAbsScaling
- Scaling to quantiles and median - RobustScaling
- Normalization to vector unit length

In this example, we will discuss Mean Normalisation.

=================================================================

Mean Normalisation
- Mean normalisation involves centering the variable at zero, and re-scaling to the value range. 
- The procedure involves subtracting the mean of each observation and then dividing by difference between the 
        minimum and maximum value:

        x_scaled = (x - x_mean) / ( x_max - x_min)

- The result of the above transformation is a distribution that is centered at 0, and its minimum and maximum values 
        are within the range of -1 to 1. 
- The shape of a mean normalised distribution will be very similar to the original distribution of the variable, 
        but the variance may change, so not identical.

- Again, this technique will not normalize the distribution of the data thus if this is the desired outcome, we 
        should implement any of the techniques discussed in section 0.5.

In a nutshell, mean normalisation:
- centers the mean at 0
- variance will be different
- may alter the shape of the original distribution
- the minimum and maximum values squeezed between -1 and 1
- preserves outliers
- Good for algorithms that require features centered at zero.

In this example
We will perform mean normalisation using the Boston House Prices data set that comes with Scikit-learn

There is no Scikit-learn transformer for mean normalisation, but we can implement it using a combination of 2 
other transformers that I will discuss in detail in the next example. We will also implement it manually with pandas. """

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# dataset for the demo
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# the scaler - for mean normalisation
from sklearn.preprocessing import StandardScaler, RobustScaler
# load the the Boston House price data

boston_dataset = load_boston()

# create a dataframe with the independent variables
data = pd.DataFrame(boston_dataset.data,
                      columns=boston_dataset.feature_names)

# add target
data['MEDV'] = boston_dataset.target

data.head()
""" 
 """

# Information about the boston house prince dataset
# you will find details about the different variables

# the aim is to predict the "Median value of the houses"
# MEDV column in this dataset

# and there are variables with characteristics about
# the homes and the neighborhoods

# print the dataset description
print(boston_dataset.DESCR)
""" .. _boston_dataset:

Boston house prices dataset
---------------------------

**Data Set Characteristics:**  

    :Number of Instances: 506 

    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

    :Missing Attribute Values: None

    :Creator: Harrison, D. and Rubinfeld, D.L.

This is a copy of UCI ML housing dataset.
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/


This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
prices and the demand for clean air', J. Environ. Economics & Management,
vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
...', Wiley, 1980.   N.B. Various transformations are used in the table on
pages 244-261 of the latter.

The Boston house-price data has been used in many machine learning papers that address regression
problems.   
     
.. topic:: References

   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann. """

# let's have a look at the main statistical parameters of the variables
# to get an idea of the feature magnitudes

data.describe()
""" 


The different variables present different value ranges, mean, max, min, standard deviations, etc. In other words, they show different magnitudes or scales. Note for this demo, how the mean values are not centered at zero, and the min and max value vary across a big range.

When performing mean normalisation on the data set, we need to first identify the mean and minimum and maximum values of the variables. These parameters need to be learned from the train set, stored, and then used to scale test and future data. Thus, we will first divide the data set into train and test, as we have done throughout the course.
"""

# let's separate the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(data.drop('MEDV', axis=1),
                                                    data['MEDV'],
                                                    test_size=0.3,
                                                    random_state=0)

X_train.shape, X_test.shape
# ((354, 13), (152, 13))

""" Mean Normalisation with pandas """
# let's first learn the mean from the train set

means = X_train.mean(axis=0)

means
""" 
CRIM         3.358284
ZN          11.809322
INDUS       11.078757
CHAS         0.064972
NOX          0.556098
RM           6.308427
AGE         68.994068
DIS          3.762459
RAD          9.353107
TAX        401.782486
PTRATIO     18.473446
B          360.601186
LSTAT       12.440650
dtype: float64 """

# let's now learn the min and max values, and the value range 
# from the train set

ranges = X_train.max(axis=0)-X_train.min(axis=0)

ranges
""" CRIM        88.96988
ZN         100.00000
INDUS       27.28000
CHAS         1.00000
NOX          0.48600
RM           5.21900
AGE         97.10000
DIS         10.95230
RAD         23.00000
TAX        524.00000
PTRATIO      9.40000
B          396.58000
LSTAT       35.25000
dtype: float64 """

# now we are ready to perform mean normalisation:
X_train_scaled = (X_train - means) / ranges
X_test_scaled = (X_test - means) / ranges

# let's have a look at the original training dataset: mean and min, max values
# I use np.round to reduce the number of decimals to 1.

np.round(X_train.describe(), 1)
""" 
 """

# let's have a look at the scaled training dataset:  mean and min, max values
# I use np.round to reduce the number of decimals to 1.

np.round(X_train_scaled.describe(), 1)
""" 


- As expected, the mean of each variable, which were not centered at zero, is now around zero and the min and max 
        values vary approximately between -1 and 1. 
- Note however, that the standard deviations vary according to how spread the variable was to begin with and is highly 
        influenced by the presence of outliers.
"""

# let's compare the variable distributions before and after scaling

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
sns.kdeplot(X_train['RM'], ax=ax1)
sns.kdeplot(X_train['LSTAT'], ax=ax1)
sns.kdeplot(X_train['CRIM'], ax=ax1)

# after scaling
ax2.set_title('After Mean Normalisation')
sns.kdeplot(X_train_scaled['RM'], ax=ax2)
sns.kdeplot(X_train_scaled['LSTAT'], ax=ax2)
sns.kdeplot(X_train_scaled['CRIM'], ax=ax2)
plt.show()

""" 
- As we can see the main effect of mean normalisation was to center all the distributions at zero, and the values 
        vary between -1 and 1. """

# let's compare the variable distributions before and after scaling

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
sns.kdeplot(X_train['AGE'], ax=ax1)
sns.kdeplot(X_train['DIS'], ax=ax1)
sns.kdeplot(X_train['NOX'], ax=ax1)

# after scaling
ax2.set_title('After Mean Normalisation')
sns.kdeplot(X_train_scaled['AGE'], ax=ax2)
sns.kdeplot(X_train_scaled['DIS'], ax=ax2)
sns.kdeplot(X_train_scaled['NOX'], ax=ax2)
plt.show()

""" 
- Compare these plots, with those derived by standardisation in the previous notebook to better understand how these 
        procedures are not identical.

Mean Normalisation with Scikit-learn: work-around
- We can implement mean normalisation by combining the use of 2 transformers. 
- A bit dirty, if you ask me, but if you are desperate to implement this technique with sklearn, this could be a way 
        forward.
"""

# set up the StandardScaler so that it removes the mean
# but does not divide by the standard deviation
scaler_mean = StandardScaler(with_mean=True, with_std=False)

# set up the robustscaler so that it does NOT remove the median
# but normalises by max()-min(), important for this to set up the
# quantile range to 0 and 100, which represent the min and max values
scaler_minmax = RobustScaler(with_centering=False,
                             with_scaling=True,
                             quantile_range=(0, 100))

# fit the scalers to the train set, it will learn the parameters
scaler_mean.fit(X_train)
scaler_minmax.fit(X_train)

# transform train and test sets
X_train_scaled = scaler_minmax.transform(scaler_mean.transform(X_train))
X_test_scaled = scaler_minmax.transform(scaler_mean.transform(X_test))

# let's transform the returned NumPy arrays to dataframes for the rest of the example

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

np.round(X_train_scaled.describe(), 1)
""" 


- See how this output is identical to the above output, where we did the scaling manually. 

- That is all for this example. I hope you enjoyed the info, and see you in the next one."""