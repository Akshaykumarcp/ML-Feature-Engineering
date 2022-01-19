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

In this example, we will discuss Scaling to vector unit length.

=================================================================

Scaling to vector unit length / unit norm
- In this procedure we scale the components of a feature vector such that the complete vector has a length of 1 or, 
        in other words a norm of 1. 
- Note that this normalisation procedure normalises the feature vector, and not the observation vector. 
- So we divide by the norm of the feature vector, observation per observation, across the different variables, 
        and not by the norm of the observation vector, across observations for the same feature.

First, let me give you the formulas, and then I illustrate with an example.

Scaling to unit norm, formulas
- Scaling to unit norm is achieved by dividing each feature vector by either the Manhattan distance (l1 norm) or 
        the Euclidean distance of the vector (l2 norm):

        X_scaled_l1 = X / l1(X)

        X_scaled_l2 X / l2(X)

        The Manhattan distance is given by the sum of the absolute components of the vector:

        l1(X) = |x1| + |x2| + ... + |xn|

        Whereas the Euclidean distance is given by the square root of the square sum of the component of the vector:

        l2(X) = sqr( x1^2 + x2^2 + ... + xn^2 )

        In the above example, x1 is variable 1, x2 is variable 2, and xn is variable n, and X is the data for 1 
                observation across variables (a row in other words).

        Note as well that as the euclidean distance squares the values of the feature vector components, outliers 
                have a heavier weight. With outliers, we may prefer to use l1 normalisation.

Scaling to unit norm, examples
    For example, if our data has 1 observations (1 row) and 3 variables:
        - number of pets
        - number of children
        - age

    The values for each variable for that single observation are 10, 15 and 20. Our vector X = [10, 15, 20]. Then:

        l1(X) = 10 + 15 + 20 = 45

        l2(X) = sqr( 10^2 + 15^2 + 20^2) = sqr( 100 + 225 + 400) = 26.9

    - The euclidean distance is always smaller than the manhattan distance.

    The normalised vector values are therefore:

        X_scaled_l1 = [ 10/45, 15/45, 20/45 ] = [0.22, 0.33, 0.44]

        X_scaled_l2 = [10/26.9, 15/26.9, 20/26.9 ] = [0.37, 0.55, 0.74]

- Scikit-learn recommends this scaling procedures for text classification or clustering. 
- For example, they quote the dot product of two l2-normalized TF-IDF vectors is the cosine similarity of the 
        vectors and is the base similarity metric for the Vector Space Model commonly used by the Information Retrieval
        community.

In this example
We will perform scaling to unit length using the Boston House Prices data set that comes with Scikit-learn """


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# the scaler - for robust scaling
from sklearn.preprocessing import Normalizer

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

""" Scaling to l1 """
# set up the scaler
scaler = Normalizer(norm='l1')

# fit the scaler, this procedure does NOTHING
scaler.fit(X_train)

# transform train and test sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# let's calculate the norm for each observation (feature vector)
# original data

np.round( np.linalg.norm(X_train, ord=1, axis=1), 1)
""" array([1024.1,  744. , 1003.5,  858. ,  732.4,  859.3,  845.9,  847. ,
        888.1,  826.7,  705.9,  808.9,  734. ,  784.6,  785.1,  819.6,
       1212.9, 1258. ,  757.6, 1223. , 1236.1,  890.2, 1166.8, 1270. ,
        919. ,  925.9, 1197. ,  770.3, 1118. ,  886.7,  719. ,  952. ,
        933.4,  745. ,  805.6,  825.1,  678.5,  947.7,  910. ,  816. ,
        769.1,  832.4,  763.5,  777. ,  850.4, 1137.7,  802.1,  703.4,
        802.6,  688. ,  801.8,  877.3,  753.8,  912.4,  936.7,  813.6,
       1194. ,  771.9,  739.5,  825.9,  890.8,  891.6,  844.4,  927.1,
        998.6,  903.3, 1250.3,  739.5, 1095.3,  857.7,  834.8,  738.4,
        855.1, 1238.4,  828.6,  958.4,  775. ,  923.4,  859.6,  830.7,
        782.3,  796.8,  953.2,  857.9, 1249.9,  716.7,  803.9,  978.8,
       1186.9,  904. , 1238.2,  981.6,  884.9,  924.9,  694.1,  725.2,
        832.4,  717.7,  917. , 1263.9,  836.7,  813.5,  992.5,  823.4,
       1224.3,  712.1,  841.2,  950.3,  702.6,  791.9,  830.4,  742.1,
        793.8, 1110.5,  832.1, 1188.5,  837.4,  846.7,  809.5,  764.6,
       1189. ,  770.4,  651.4,  717.8, 1265.8, 1004. ,  782.1,  922.7,
        871.4,  860. ,  860.1,  766.5, 1259.6,  920.1,  708.2,  779.3,
        785.7,  815. ,  825. ,  856.5,  784.2,  834.9, 1003.2,  963.2,
        924.1, 1141.8, 1192.2,  777.5,  767.7, 1197.4,  887.2,  839.3,
        819.9,  818.9,  822.7, 1006.2,  926.2,  748.9,  920.8,  790.7,
        810.2, 1235.9,  889. ,  921.2,  895.3,  774.9,  924.8,  926. ,
       1248.6, 1006.3,  817.5,  800.7, 1227.4,  831.4,  787.4, 1312. ,
        832.1,  769.5, 1252.1,  822.7,  884. ,  787.2,  749.4,  827.8,
        962.2,  702.8,  745.4,  731.7,  727.8,  743.7,  888.3,  720.2,
       1332.3, 1185.2,  776.4,  784.6, 1255.2,  717.8, 1239. ,  782.1,
        767.9,  807.6,  857.5, 1265.4,  737.9,  800. ,  769.3, 1231.7,
        783.1,  667. ,  726.4,  758.5,  841.3, 1243.1, 1268.4, 1174.9,
        736. ,  828.9,  827.5, 1270. , 1000.2,  919.4,  866. ,  681.7,
       1249. ,  810.9,  901.7, 1218.9,  928.7,  744.7,  843.4,  736.1,
        897.6,  804.6, 1219.5,  794.5,  830.4, 1241.1,  732. , 1211.2,
        890.6,  838.1,  753.5,  795.8,  801.3, 1277.8, 1256. ,  801.3,
        704.7,  808.2,  931.3,  977.4,  702.4,  713.7, 1240.1, 1001.4,
        867.2,  806. ,  802.1,  774.1,  783.6,  739.2,  719.6, 1276.7,
       1264.7,  865.4, 1104.8, 1162.6,  778.9,  781.8,  776.8,  945.3,
       1234.4,  794.1,  830.3,  880.1,  788.6,  799.5, 1231.8, 1236.9,
        669.1,  934.3, 1253.3, 1239.6,  734.5,  855.2,  753.4, 1260.9,
        730.5, 1182.1, 1253. , 1004.4,  793.7,  847.5,  724.4, 1237.2,
        941.6,  906.5,  859.6,  877.7,  842. ,  694.3,  997.8,  695.3,
       1251.1, 1167. ,  837.6,  855. , 1301.8, 1238.6,  968.8,  822.6,
       1216.7,  807.1,  775.9,  707.8, 1260.5,  943.2,  706.4,  839.5,
        759.2, 1221.5,  923.8, 1219.9,  780.6,  755.1,  701.2,  757.2,
        797.2,  731.4, 1237.1,  802.4, 1265.4,  785.5,  766.3,  839.7,
       1255.2,  792.9, 1224.5,  748.3,  743.8,  790.3,  808.3,  760.4,
        828.9,  854.6, 1226.7,  797. ,  788. ,  807.4,  900.5,  962.7,
        770.2,  830.6]) """

# let's calculate the norm for each observation (feature vector) scaled data
np.round( np.linalg.norm(X_train_scaled, ord=1, axis=1), 1)
""" array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

Now, each feature vector has a l1 norm of 1.
"""

# let's transform the returned NumPy arrays to dataframes for the rest of the example
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# let's look at the individual values of the variables in the original data
X_train.describe()
""" 
             CRIM          ZN       INDUS        CHAS         NOX          RM         AGE         DIS         RAD         TAX     PTRATIO           B       LSTAT
count  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000
mean     3.358284   11.809322   11.078757    0.064972    0.556098    6.308427   68.994068    3.762459    9.353107  401.782486   18.473446  360.601186   12.440650
std      8.353223   23.653056    6.993821    0.246825    0.115601    0.702009   28.038429    2.067661    8.671999  170.592404    2.224809   85.621945    7.078485
min      0.006320    0.000000    0.460000    0.000000    0.385000    3.561000    2.900000    1.174200    1.000000  187.000000   12.600000    0.320000    1.730000
25%      0.073425    0.000000    4.950000    0.000000    0.450000    5.895250   45.175000    2.107650    4.000000  276.000000   17.400000  376.057500    6.735000
50%      0.262660    0.000000    8.560000    0.000000    0.538000    6.215500   79.450000    3.215700    5.000000  311.000000   19.100000  391.605000   11.160000
75%      3.103700   20.000000   18.100000    0.000000    0.629250    6.647250   93.750000    5.079300   24.000000  666.000000   20.200000  395.690000   16.717500
max     88.976200  100.000000   27.740000    1.000000    0.871000    8.780000  100.000000   12.126500   24.000000  711.000000   22.000000  396.900000   36.980000 """

# let's look at the individual values of the variables in the original data
X_train_scaled.describe()
""" 
             CRIM          ZN       INDUS        CHAS         NOX          RM         AGE         DIS         RAD         TAX     PTRATIO           B       LSTAT
count  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000
mean     0.037675    0.118093    0.389251    0.064972    0.352054    0.526428    0.680680    0.236321    0.363179    0.409890    0.624835    0.908470    0.303848
std      0.093888    0.236531    0.256372    0.246825    0.237861    0.134510    0.288758    0.188788    0.377043    0.325558    0.236682    0.215901    0.200808
min      0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
25%      0.000754    0.000000    0.164589    0.000000    0.133745    0.447260    0.435376    0.085229    0.130435    0.169847    0.510638    0.947444    0.141986
50%      0.002881    0.000000    0.296921    0.000000    0.314815    0.508622    0.788363    0.186399    0.173913    0.236641    0.691489    0.986648    0.267518
75%      0.034814    0.200000    0.646628    0.000000    0.502572    0.591349    0.935633    0.356555    1.000000    0.914122    0.808511    0.996949    0.425177
max      1.000000    1.000000    1.000000    1.000000    1.000000    1.000000    1.000000    1.000000    1.000000    1.000000    1.000000    1.000000    1.000000

The values have been squeezed in a smaller value range.  """

# let's compare the variable distributions before and after scaling
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
sns.kdeplot(X_train['RM'], ax=ax1)
sns.kdeplot(X_train['LSTAT'], ax=ax1)
sns.kdeplot(X_train['CRIM'], ax=ax1)

# after scaling
ax2.set_title('After Unit Norm Scaling')
sns.kdeplot(X_train_scaled['RM'], ax=ax2)
sns.kdeplot(X_train_scaled['LSTAT'], ax=ax2)
sns.kdeplot(X_train_scaled['CRIM'], ax=ax2)
plt.show()

# let's compare the variable distributions before and after scaling
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
#sns.kdeplot(X_train['AGE'], ax=ax1)
#sns.kdeplot(X_train['DIS'], ax=ax1)
sns.kdeplot(X_train['NOX'], ax=ax1)

# after scaling
ax2.set_title('After Unit Norm Scaling')
#sns.kdeplot(X_train_scaled['AGE'], ax=ax2)
#sns.kdeplot(X_train_scaled['DIS'], ax=ax2)
sns.kdeplot(X_train_scaled['NOX'], ax=ax2)
plt.show()

""" 
- See how this normalisation changes the distribution of the original variable quite dramatically.
- Go ahead and comment in an out the different variables to have a better look.

Scaling to l2 """
# set up the scaler
scaler = Normalizer(norm='l2')

# fit the scaler, this procedure does NOTHING
scaler.fit(X_train)

# transform train and test sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# let's calculate the norm for each observation (feature vector)
# original data

np.round( np.linalg.norm(X_train, ord=2, axis=1), 1)
""" array([600.5, 458.4, 598. , 519.1, 434. , 511. , 527.9, 496.7, 675.2,
       496.5, 450.2, 492.1, 453.5, 492.4, 492.9, 491.5, 770.1, 782.2,
       465.2, 759.8, 778.6, 535.3, 764.4, 781.5, 561.9, 562.8, 768.1,
       468.2, 725.5, 673.1, 458.8, 572.5, 559.8, 488. , 490.2, 488.9,
       441.3, 680.4, 675.6, 483.9, 483.9, 506.2, 474.8, 481.9, 517.8,
       747.8, 503.3, 438.2, 483. , 424. , 478.8, 519.9, 451.4, 546.2,
       558.3, 516.9, 773.8, 470.4, 467.5, 501.6, 674.5, 528.1, 508.2,
       677.1, 593.2, 556.5, 781.9, 465.9, 707.6, 509.6, 501.3, 449.4,
       512.5, 778.8, 483.6, 587.7, 485.4, 558.1, 673.1, 504.6, 494.9,
       509. , 589.2, 510.6, 777. , 439.2, 478.8, 593.3, 760. , 534.5,
       779.9, 586.6, 551.3, 558.4, 446.2, 418.8, 496. , 407.8, 676.2,
       782.8, 671. , 499. , 593.6, 482.6, 779.2, 456.1, 483.4, 573.4,
       456.5, 477.3, 493.3, 485.7, 490.5, 716.1, 497.2, 760.6, 524.9,
       672.1, 488.2, 443.7, 751. , 492.5, 424.3, 458.7, 783. , 596.4,
       482.9, 557.2, 515.4, 523.3, 510.3, 483.8, 782.2, 565.5, 447.2,
       461.6, 478.5, 483.8, 526.5, 509.4, 484.1, 508.7, 599. , 591.9,
       555.7, 733.4, 752.5, 475.7, 474.5, 759.2, 563.3, 502.8, 503.7,
       506.2, 497.6, 682.7, 563.5, 487.4, 558.2, 484.7, 490.9, 774.6,
       553.2, 544.4, 537.4, 495.4, 548.7, 556.5, 770. , 599.5, 502. ,
       481.1, 774.1, 505.8, 483.4, 779.9, 502.8, 475.3, 781.2, 502.6,
       576.5, 491. , 455.7, 502.3, 681.5, 455.6, 439. , 439. , 449. ,
       427.5, 557.6, 445.8, 786.8, 775.3, 475.3, 493.4, 782.2, 448.9,
       772.3, 488.6, 461.9, 476.8, 511.7, 782.7, 469.5, 474.6, 474.1,
       778.3, 485.2, 450.9, 451.2, 467.6, 495.3, 772.6, 783. , 746.1,
       428.5, 503.2, 494. , 782.9, 598.3, 563.6, 510.8, 458.7, 778.2,
       500. , 533.7, 772. , 678.4, 482.5, 517.3, 455.6, 560.4, 496.7,
       771.5, 467.4, 487.8, 780.8, 456.2, 765.8, 530.3, 527.6, 457.2,
       471.5, 490.4, 783.4, 780.8, 484.3, 453.8, 489.5, 547.8, 575.6,
       451.5, 448.5, 773. , 599.3, 674. , 487.8, 478. , 492. , 463.6,
       470.5, 431.7, 819.6, 779.5, 673. , 720. , 732.3, 483.4, 488.5,
       472. , 587.2, 780. , 474.6, 481.6, 674.5, 493.4, 479.4, 765.6,
       775.9, 449.2, 559.4, 771.4, 780.9, 453.9, 673.7, 483.9, 782.8,
       466.9, 746.2, 777.8, 599.6, 474.6, 504.4, 452. , 779.9, 568.2,
       538.5, 504.7, 533.7, 496.9, 395.5, 596. , 438.6, 781.7, 740. ,
       498.7, 672.9, 784.1, 780.8, 571.4, 500.2, 786.5, 499.9, 487.9,
       457.3, 782.7, 678.9, 457.7, 491.8, 481.5, 779.1, 560.5, 750.5,
       477.9, 496.5, 451.7, 441. , 479.1, 458.5, 797.1, 497.4, 819.4,
       476. , 478.6, 507.2, 782.3, 488.2, 778.1, 490.8, 470. , 474.3,
       482.9, 471.3, 491.8, 504.5, 777.4, 477.7, 502.2, 491.5, 560.3,
       591.3, 465.4, 503.5]) """

# let's calculate the norm for each observation (feature vector) scaledl data

np.round( np.linalg.norm(X_train_scaled, ord=2, axis=1), 1)
""" array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]) """

# let's transform the returned NumPy arrays to dataframes for the rest of the example
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

X_train.describe()
""" 
             CRIM          ZN       INDUS        CHAS         NOX          RM         AGE         DIS         RAD         TAX     PTRATIO           B       LSTAT
count  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000
mean     3.358284   11.809322   11.078757    0.064972    0.556098    6.308427   68.994068    3.762459    9.353107  401.782486   18.473446  360.601186   12.440650
std      8.353223   23.653056    6.993821    0.246825    0.115601    0.702009   28.038429    2.067661    8.671999  170.592404    2.224809   85.621945    7.078485
min      0.006320    0.000000    0.460000    0.000000    0.385000    3.561000    2.900000    1.174200    1.000000  187.000000   12.600000    0.320000    1.730000
25%      0.073425    0.000000    4.950000    0.000000    0.450000    5.895250   45.175000    2.107650    4.000000  276.000000   17.400000  376.057500    6.735000
50%      0.262660    0.000000    8.560000    0.000000    0.538000    6.215500   79.450000    3.215700    5.000000  311.000000   19.100000  391.605000   11.160000
75%      3.103700   20.000000   18.100000    0.000000    0.629250    6.647250   93.750000    5.079300   24.000000  666.000000   20.200000  395.690000   16.717500
max     88.976200  100.000000   27.740000    1.000000    0.871000    8.780000  100.000000   12.126500   24.000000  711.000000   22.000000  396.900000   36.980000 """

X_train_scaled.describe()
""" 
             CRIM          ZN       INDUS        CHAS         NOX          RM         AGE         DIS         RAD         TAX     PTRATIO           B       LSTAT
count  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000  354.000000
mean     0.037675    0.118093    0.389251    0.064972    0.352054    0.526428    0.680680    0.236321    0.363179    0.409890    0.624835    0.908470    0.303848
std      0.093888    0.236531    0.256372    0.246825    0.237861    0.134510    0.288758    0.188788    0.377043    0.325558    0.236682    0.215901    0.200808
min      0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
25%      0.000754    0.000000    0.164589    0.000000    0.133745    0.447260    0.435376    0.085229    0.130435    0.169847    0.510638    0.947444    0.141986
50%      0.002881    0.000000    0.296921    0.000000    0.314815    0.508622    0.788363    0.186399    0.173913    0.236641    0.691489    0.986648    0.267518
75%      0.034814    0.200000    0.646628    0.000000    0.502572    0.591349    0.935633    0.356555    1.000000    0.914122    0.808511    0.996949    0.425177
max      1.000000    1.000000    1.000000    1.000000    1.000000    1.000000    1.000000    1.000000    1.000000    1.000000    1.000000    1.000000    1.000000 """

# let's compare the variable distributions before and after scaling
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
sns.kdeplot(X_train['RM'], ax=ax1)
sns.kdeplot(X_train['LSTAT'], ax=ax1)
sns.kdeplot(X_train['CRIM'], ax=ax1)

# after scaling
ax2.set_title('After Unit Norm Scaling')
sns.kdeplot(X_train_scaled['RM'], ax=ax2)
sns.kdeplot(X_train_scaled['LSTAT'], ax=ax2)
sns.kdeplot(X_train_scaled['CRIM'], ax=ax2)
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
#sns.kdeplot(X_train['AGE'], ax=ax1)
#sns.kdeplot(X_train['DIS'], ax=ax1)
sns.kdeplot(X_train['NOX'], ax=ax1)

# after scaling
ax2.set_title('After Unit Norm Scaling')
#sns.kdeplot(X_train_scaled['AGE'], ax=ax2)
#sns.kdeplot(X_train_scaled['DIS'], ax=ax2)
sns.kdeplot(X_train_scaled['NOX'], ax=ax2)
plt.show()

# That is all for this example. I hope you enjoyed the info, and see you in the next one.