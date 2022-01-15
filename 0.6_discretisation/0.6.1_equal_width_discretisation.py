""" 
Discretisation
- Discretisation is the process of transforming continuous variables into discrete variables by creating a set of 
        contiguous intervals that span the range of the variable's values. 
- Discretisation is also called binning, where bin is an alternative name for interval.

- Discretisation helps handle outliers and may improve value spread in skewed variables
- Discretisation helps handle outliers by placing these values into the lower or higher intervals, 
        together with the remaining inlier values of the distribution. 
- Thus, these outlier observations no longer differ from the rest of the values at the tails of the distribution,
        as they are now all together in the same interval / bucket. 
- In addition, by creating appropriate bins or intervals, discretisation can help spread the values of a skewed 
        variable across a set of bins with equal number of observations.

Discretisation approaches
- There are several approaches to transform continuous variables into discrete ones. 
- Discretisation methods fall into 2 categories: 
    - supervised and 
        Supervised methods typically use target information in order to create the bins or intervals.
    - unsupervised. 
        Unsupervised methods do not use any information, other than the variable distribution, 
        to create the contiguous bins in which the values will be placed. 

Unsupervised discretisation methods
- Equal width discretisation
- Equal frequency discretisation
- K-means discretisation

Supervised discretisation methods
- Discretisation using decision trees

In this example, I will describe equal width discretisation.

Equal width discretisation
- Equal width discretisation divides the scope of possible values into N bins of the same width.
- The width is determined by the range of values in the variable and the number of bins we wish to use to divide 
    the variable:

        width = (max value - min value) / N

        where N is the number of bins or intervals.

        - For example if the values of the variable vary between 0 and 100, we create 5 bins like this: 
            width = (100-0) / 5 = 20.
            The bins thus are 0-20, 20-40, 40-60, 80-100. 
            The first and final bins (0-20 and 80-100) can be expanded to accommodate outliers 
            (that is, values under 0 or greater than 100 would be placed in those bins as well).

        - There is no rule of thumb to define N, that is something to determine experimentally.

In this example:
We will learn how to perform equal width binning using the Titanic dataset with 
- pandas and NumPy
- Scikit-learn """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

# load the numerical variables of the Titanic Dataset
data = pd.read_csv('dataset/titanic.csv',usecols=['age', 'fare', 'survived'])

data.head()
"""    survived      age      fare
0         1  29.0000  211.3375
1         1   0.9167  151.5500
2         0   2.0000  151.5500
3         0  30.0000  151.5500
4         0  25.0000  151.5500 """

# Let's separate into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    data[['age', 'fare']],
    data['survived'],
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape
# ((916, 2), (393, 2))

# The variables Age and fare contain missing data, that I will fill by extracting a random sample of the variable.

def impute_na(data, variable):
    df = data.copy()
    # random sampling
    df[variable + '_random'] = df[variable]
    # extract the random sample to fill the na
    random_sample = X_train[variable].dropna().sample(
        df[variable].isnull().sum(), random_state=0)
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(), variable + '_random'] = random_sample
    return df[variable + '_random']

# replace NA in both  train and test sets
X_train['age'] = impute_na(data, 'age')
X_test['age'] = impute_na(data, 'age')

X_train['fare'] = impute_na(data, 'fare')
X_test['fare'] = impute_na(data, 'fare')

# let's explore the distribution of age and fare
data[['age', 'fare']].hist(bins=30, figsize=(8,4))
plt.show()

""" 
Equal width discretisation with pandas and NumPy

First we need to determine the intervals edges or limits.
 """

# let's capture the range of the variable age
age_range = X_train['age'].max() - X_train['age'].min()

age_range
# 73.8333

# let's divide the range into 10 equal width bins
age_range / 10
# 7.383329999999999

# The range or width of our intervals will be 7 years.

# now let's capture the lower and upper boundaries
min_value = int(np.floor( X_train['age'].min()))
max_value = int(np.ceil( X_train['age'].max()))

min_value
# 0

max_value
# 74

# let's round the bin width
inter_value = int(np.round(age_range / 10))

min_value, max_value, inter_value
# (0, 74, 7)

# let's capture the interval limits, so we can pass them to the pandas cut 
# function to generate the bins

intervals = [i for i in range(min_value, max_value+inter_value, inter_value)]

intervals
# [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77]

# let's make labels to label the different bins
labels = ['Bin_' + str(i) for i in range(1, len(intervals))]

labels
""" ['Bin_1',
 'Bin_2',
 'Bin_3',
 'Bin_4',
 'Bin_5',
 'Bin_6',
 'Bin_7',
 'Bin_8',
 'Bin_9',
 'Bin_10',
 'Bin_11'] """

# create binned age / discretise age

# create one column with labels
X_train['Age_disc_labels'] = pd.cut(x=X_train['age'],
                                    bins=intervals,
                                    labels=labels,
                                    include_lowest=True)

# and one with bin boundaries
X_train['Age_disc'] = pd.cut(x=X_train['age'],
                             bins=intervals,
                             include_lowest=True)

X_train.head(10)
""" 
       age      fare Age_disc_labels       Age_disc
501   13.0   19.5000           Bin_2    (7.0, 14.0]
588    4.0   23.0000           Bin_1  (-0.001, 7.0]
402   30.0   13.8583           Bin_5   (28.0, 35.0]
1193  21.0    7.7250           Bin_3   (14.0, 21.0]
686   22.0    7.7250           Bin_4   (21.0, 28.0]
971   16.0    7.8792           Bin_3   (14.0, 21.0]
117   30.0   56.9292           Bin_5   (28.0, 35.0]
540    2.0   26.0000           Bin_1  (-0.001, 7.0]
294   49.0  110.8833           Bin_7   (42.0, 49.0]
261   35.0   26.2875           Bin_5   (28.0, 35.0] 

- We can see in the above output how by discretising using equal width, we placed each Age observation within one 
        interval / bin. 
        For example, age=13 was placed in the 7-14 interval, whereas age 30 was placed into the 28-35 interval.

- When performing equal width discretisation, we guarantee that the intervals are all of the same length, 
        however there won't necessarily be the same number of observations in each of the intervals. See below: """

X_train.groupby('Age_disc')['age'].count()
""" Age_disc
(-0.001, 7.0]     48
(7.0, 14.0]       31
(14.0, 21.0]     155
(21.0, 28.0]     230
(28.0, 35.0]     176
(35.0, 42.0]     105
(42.0, 49.0]      75
(49.0, 56.0]      51
(56.0, 63.0]      32
(63.0, 70.0]      10
(70.0, 77.0]       3
Name: age, dtype: int64 """

X_train.groupby('Age_disc')['age'].count().plot.bar()
plt.xticks(rotation=45)
plt.ylabel('Number of observations per bin')
plt.show()

""" 
- The majority of people on the Titanic were between 14-42 years of age.
- Now, we can discretise Age in the test set, using the same interval boundaries that we calculated for the train set:
 """
X_test['Age_disc_labels'] = pd.cut(x=X_test['age'],
                                   bins=intervals,
                                   labels=labels,
                                   include_lowest=True)

X_test['Age_disc'] = pd.cut(x=X_test['age'],
                            bins=intervals,
                            include_lowest=True)

X_test.head()
""" 
       age     fare Age_disc_labels      Age_disc
1139  38.0   7.8958           Bin_6  (35.0, 42.0]
533   21.0  21.0000           Bin_3  (14.0, 21.0]
459   42.0  27.0000           Bin_6  (35.0, 42.0]
1150  34.0  14.5000           Bin_5  (28.0, 35.0]
393   25.0  31.5000           Bin_4  (21.0, 28.0] """

# if the distributions in train and test set are similar, we should expect similar propotion of
# observations in the different intervals in the train and test set
# let's see that below

t1 = X_train.groupby(['Age_disc'])['age'].count() / len(X_train)
t2 = X_test.groupby(['Age_disc'])['age'].count() / len(X_test)

tmp = pd.concat([t1, t2], axis=1)
tmp.columns = ['train', 'test']
tmp.plot.bar()
plt.xticks(rotation=45)
plt.ylabel('Number of observations per bin')
plt.show()

""" Equal width discretisation with Scikit-learn """

# Let's separate into train and test set

X_train, X_test, y_train, y_test = train_test_split(
    data[['age', 'fare']],
    data['survived'],
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape
# ((916, 2), (393, 2))

# replace NA in both  train and test sets
X_train['age'] = impute_na(data, 'age')
X_test['age'] = impute_na(data, 'age')

X_train['fare'] = impute_na(data, 'fare')
X_test['fare'] = impute_na(data, 'fare')

disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')

disc.fit(X_train[['age', 'fare']])
# KBinsDiscretizer(encode='ordinal', n_bins=10, strategy='uniform')

disc.bin_edges_
""" array([array([ 0.1667 ,  7.55003, 14.93336, 22.31669, 29.70002, 37.08335,
       44.46668, 51.85001, 59.23334, 66.61667, 74.     ]),
       array([  0.     ,  51.23292, 102.46584, 153.69876, 204.93168, 256.1646 ,
       307.39752, 358.63044, 409.86336, 461.09628, 512.3292 ])],
      dtype=object) """

train_t = disc.transform(X_train[['age', 'fare']])

train_t = pd.DataFrame(train_t, columns = ['age', 'fare'])

train_t.head()
""" 
   age  fare
0  1.0   0.0
1  0.0   0.0
2  4.0   0.0
3  2.0   0.0
4  2.0   0.0 """

test_t = disc.transform(X_test[['age', 'fare']])

test_t = pd.DataFrame(test_t, columns = ['age', 'fare'])
t1 = train_t.groupby(['age'])['age'].count() / len(train_t)
t2 = test_t.groupby(['age'])['age'].count() / len(test_t)

tmp = pd.concat([t1, t2], axis=1)
tmp.columns = ['train', 'test']
tmp.plot.bar()
plt.xticks(rotation=0)
plt.ylabel('Number of observations per bin')
plt.show()

t1 = train_t.groupby(['fare'])['fare'].count() / len(train_t)
t2 = test_t.groupby(['fare'])['fare'].count() / len(test_t)

tmp = pd.concat([t1, t2], axis=1)
tmp.columns = ['train', 'test']
tmp.plot.bar()
plt.xticks(rotation=0)
plt.ylabel('Number of observations per bin')
plt.show()

# distribution are similar for train and test set

# That is all for this EXAMPLE. I hope you enjoyed the information, and see you in the next one.