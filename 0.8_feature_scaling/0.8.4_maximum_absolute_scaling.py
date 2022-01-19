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

In this example, we will discuss Maximum Absolute Scaling.

=================================================================

Scaling to maximum value - MaxAbsScaling
- Maximum absolute scaling scales the data to its maximum value:

        X_scaled = X / X.max

- The result of the above transformation is a distribution which values vary within the range of -1 to 1.
- But the mean is not centered at zero and the standard deviation varies across variables.

- Scikit-learn suggests that this transformer is meant for data that is centered at zero, and for sparse data.

In a nutshell, MaxAbsScaling:
- does not center the mean at 0 (but it might be a good idea t center it with another method)
- variance varies across variables
- may not preserve the shape of the original distribution
- sensitive outliers

In this example
- We will perform scaling to the maximum value using the Boston House Prices data set that comes with Scikit-learn """

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
# the scaler - for MaxAbsScaling, with centering
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

# load the the Boston House price data

boston_dataset = load_boston()

# create a dataframe with the independent variables
data = pd.DataFrame(boston_dataset.data,
                      columns=boston_dataset.feature_names)

# add target
data['MEDV'] = boston_dataset.target

data.head()
""" 
      CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  PTRATIO       B  LSTAT  MEDV
0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0     15.3  396.90   4.98  24.0
1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0     17.8  396.90   9.14  21.6
2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0     17.8  392.83   4.03  34.7
3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0     18.7  394.63   2.94  33.4
4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0     18.7  396.90   5.33  36.2 """

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
             CRIM          ZN       INDUS        CHAS         NOX          RM         AGE         DIS         RAD         TAX     PTRATIO           B       LSTAT        MEDV
count  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000
mean     3.613524   11.363636   11.136779    0.069170    0.554695    6.284634   68.574901    3.795043    9.549407  408.237154   18.455534  356.674032   12.653063   22.532806
std      8.601545   23.322453    6.860353    0.253994    0.115878    0.702617   28.148861    2.105710    8.707259  168.537116    2.164946   91.294864    7.141062    9.197104
min      0.006320    0.000000    0.460000    0.000000    0.385000    3.561000    2.900000    1.129600    1.000000  187.000000   12.600000    0.320000    1.730000    5.000000
25%      0.082045    0.000000    5.190000    0.000000    0.449000    5.885500   45.025000    2.100175    4.000000  279.000000   17.400000  375.377500    6.950000   17.025000
50%      0.256510    0.000000    9.690000    0.000000    0.538000    6.208500   77.500000    3.207450    5.000000  330.000000   19.050000  391.440000   11.360000   21.200000
75%      3.677083   12.500000   18.100000    0.000000    0.624000    6.623500   94.075000    5.188425   24.000000  666.000000   20.200000  396.225000   16.955000   25.000000
max     88.976200  100.000000   27.740000    1.000000    0.871000    8.780000  100.000000   12.126500   24.000000  711.000000   22.000000  396.900000   37.970000   50.000000

- The different variables present different value ranges, mean, max, min, standard deviations, etc. 
- In other words, they show different magnitudes or scales. Note for this demo, how the maximum values are are quite 
        different in the different variables.

- When performing maximum absolute scaling on the data set, we need to first identify the maximum values of the variables. 
- These parameters need to be learned from the train set, stored, and then used to scale test and future data. 
- Thus, we will first divide the data set into train and test, as we have done throughout the course.
"""

# let's separate into training and testing set
X_train, X_test, y_train, y_test = train_test_split(data.drop('MEDV', axis=1),
                                                    data['MEDV'],
                                                    test_size=0.3,
                                                    random_state=0)

X_train.shape, X_test.shape
# ((354, 13), (152, 13))

""" 
MaxAbsScaling
- The MaxAbsScaler from scikit-learn re-scales features to their maximum value, so that the new maximum value is 1.
 """
# set up the scaler
scaler = MaxAbsScaler()

# fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train)

# transform train and test sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# the scaler stores the maximum values of the features as learned from train set
scaler.max_abs_
""" array([ 88.9762, 100.    ,  27.74  ,   1.    ,   0.871 ,   8.78  ,
       100.    ,  12.1265,  24.    , 711.    ,  22.    , 396.9   ,
        36.98  ]) """

# let's transform the returned NumPy arrays to dataframes for the rest of the example
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# let's have a look at the original training dataset: maximum values
# I use np.round to reduce the number of decimals to 1.

np.round(X_train.describe(), 1)
""" 
        CRIM     ZN  INDUS   CHAS    NOX     RM    AGE    DIS    RAD    TAX  PTRATIO      B  LSTAT
count  354.0  354.0  354.0  354.0  354.0  354.0  354.0  354.0  354.0  354.0    354.0  354.0  354.0
mean     3.4   11.8   11.1    0.1    0.6    6.3   69.0    3.8    9.4  401.8     18.5  360.6   12.4
std      8.4   23.7    7.0    0.2    0.1    0.7   28.0    2.1    8.7  170.6      2.2   85.6    7.1
min      0.0    0.0    0.5    0.0    0.4    3.6    2.9    1.2    1.0  187.0     12.6    0.3    1.7
25%      0.1    0.0    5.0    0.0    0.4    5.9   45.2    2.1    4.0  276.0     17.4  376.1    6.7
50%      0.3    0.0    8.6    0.0    0.5    6.2   79.4    3.2    5.0  311.0     19.1  391.6   11.2
75%      3.1   20.0   18.1    0.0    0.6    6.6   93.8    5.1   24.0  666.0     20.2  395.7   16.7
max     89.0  100.0   27.7    1.0    0.9    8.8  100.0   12.1   24.0  711.0     22.0  396.9   37.0 """

# let's have a look at the scaled training dataset: maximum values
# I use np.round to reduce the number of decimals to 1.

np.round(X_train_scaled.describe(), 1)
""" 
        CRIM     ZN  INDUS   CHAS    NOX     RM    AGE    DIS    RAD    TAX  PTRATIO      B  LSTAT
count  354.0  354.0  354.0  354.0  354.0  354.0  354.0  354.0  354.0  354.0    354.0  354.0  354.0
mean     0.0    0.1    0.4    0.1    0.4    0.5    0.7    0.2    0.4    0.4      0.6    0.9    0.3
std      0.1    0.2    0.3    0.2    0.2    0.1    0.3    0.2    0.4    0.3      0.2    0.2    0.2
min      0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0      0.0    0.0    0.0
25%      0.0    0.0    0.2    0.0    0.1    0.4    0.4    0.1    0.1    0.2      0.5    0.9    0.1
50%      0.0    0.0    0.3    0.0    0.3    0.5    0.8    0.2    0.2    0.2      0.7    1.0    0.3
75%      0.0    0.2    0.6    0.0    0.5    0.6    0.9    0.4    1.0    0.9      0.8    1.0    0.4
max      1.0    1.0    1.0    1.0    1.0    1.0    1.0    1.0    1.0    1.0      1.0    1.0    1.0

The maximum values are re-scaled to 1. Every other parameter, changes across variables.
 """

# let's compare the variable distributions before and after scaling
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
sns.kdeplot(X_train['RM'], ax=ax1)
sns.kdeplot(X_train['LSTAT'], ax=ax1)
sns.kdeplot(X_train['CRIM'], ax=ax1)

# after scaling
ax2.set_title('After Max Abs Scaling')
sns.kdeplot(X_train_scaled['RM'], ax=ax2)
sns.kdeplot(X_train_scaled['LSTAT'], ax=ax2)
sns.kdeplot(X_train_scaled['CRIM'], ax=ax2)
plt.show()

# let's compare the variable distributions before and after scaling

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
sns.kdeplot(X_train['AGE'], ax=ax1)
sns.kdeplot(X_train['DIS'], ax=ax1)
sns.kdeplot(X_train['NOX'], ax=ax1)

# after scaling
ax2.set_title('After Max Abs Scaling')
sns.kdeplot(X_train_scaled['AGE'], ax=ax2)
sns.kdeplot(X_train_scaled['DIS'], ax=ax2)
sns.kdeplot(X_train_scaled['NOX'], ax=ax2)
plt.show()

""" 
Centering + MaxAbsScaling
- We can center the distributions at zero and then scale to its absolute maximum, as recommended by Scikit-learn by 
        combining the use of 2 transformers.
 """
# set up the StandardScaler so that it removes the mean
# but does not divide by the standard deviation
scaler_mean = StandardScaler(with_mean=True, with_std=False)

# set up the MaxAbsScaler normally
scaler_maxabs = MaxAbsScaler()

# fit the scalers to the train set, it will learn the parameters
scaler_mean.fit(X_train)
scaler_maxabs.fit(X_train)

# transform train and test sets
X_train_scaled = scaler_maxabs.transform(scaler_mean.transform(X_train))
X_test_scaled = scaler_maxabs.transform(scaler_mean.transform(X_test))

# let's transform the returned NumPy arrays to dataframes for the rest of the example

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# let's compare the variable distributions before and after scaling
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
sns.kdeplot(X_train['AGE'], ax=ax1)
sns.kdeplot(X_train['DIS'], ax=ax1)
sns.kdeplot(X_train['NOX'], ax=ax1)

# after scaling
ax2.set_title('After Max Abs Scaling')
sns.kdeplot(X_train_scaled['AGE'], ax=ax2)
sns.kdeplot(X_train_scaled['DIS'], ax=ax2)
sns.kdeplot(X_train_scaled['NOX'], ax=ax2)
plt.show()

# Similar effect to mean normalisation, but in this case, the values vary across a narrower range.

# That is all for this example. I hope you enjoyed the info, and see you in the next one.