""" 
Discretisation plus Encoding
- What shall we do with the variable after discretisation? 
- should we use the buckets as a numerical variable? or 
- should we use the intervals as categorical variable?

The answer is, you can do either.
- If you are building decision tree based algorithms and the output of the discretisation are integers
         (each integer referring to a bin), then you can use those directly, as decision trees will pick up non-linear 
         relationships between the discretised variable and the target.

- If you are building linear models instead, the bins may not necessarily hold a linear relationship with the target.
         In this case, it may help improve model performance to treat the bins as categories and to one hot encoding, 
         or target guided encodings like mean encoding, weight of evidence, or target guided ordinal encoding.

We can easily do so by 

In this example
- We will perform equal frequency discretisation followed by target guided orginal encoding using the titanic dataset

If instead you would like to do weight of evidence or mean target encoding, you need only replace the ."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

# load the numerical variables of the Titanic Dataset
data = pd.read_csv('dataset/titanic.csv',usecols=['age', 'fare', 'survived'])

data.head()
""" 
   survived      age      fare
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

# The variables Age and Fare contain missing data, that I will fill by extracting a random sample of the variable.

def impute_na(data, variable):
    # function to fill NA with a random sample
    df = data.copy()
    # random sampling
    df[variable+'_random'] = df[variable]
    # extract the random sample to fill the na
    random_sample = X_train[variable].dropna().sample(
        df[variable].isnull().sum(), random_state=0)
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(), variable+'_random'] = random_sample
    return df[variable+'_random']

# replace NA in both train and test sets
X_train['age'] = impute_na(data, 'age')
X_test['age'] = impute_na(data, 'age')

X_train['fare'] = impute_na(data, 'fare')
X_test['fare'] = impute_na(data, 'fare')

# let's explore the distribution of age

X_train[['age', 'fare']].hist(bins=30, figsize=(8,4))
plt.show()


""" Equal frequency discretisation with Scikit-learn """

disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

disc.fit(X_train[['age', 'fare']])
# KBinsDiscretizer(encode='ordinal', n_bins=10, strategy='quantile')

disc.bin_edges_
""" array([array([ 0.1667, 16.    , 20.    , 22.25  , 25.    , 28.    , 31.    ,
       36.    , 42.    , 50.    , 74.    ]),
       array([  0.    ,   7.55  ,   7.7958,   8.05  ,  10.5   ,  14.4542,
        21.075 ,  26.55  ,  40.125 ,  79.025 , 512.3292])], dtype=object) """

train_t = disc.transform(X_train[['age', 'fare']])

train_t = pd.DataFrame(train_t, columns = ['age', 'fare'])

train_t.head()
""" 
   age  fare
0  0.0   5.0
1  0.0   6.0
2  5.0   4.0
3  2.0   1.0
4  2.0   1.0 """
""" 
- We can see that the top intervals have less observations. 
- This may happen with skewed distributions if we try to divide in a high number of intervals. 
- To make the value spread more homogeneous, we should discretise in less intervals.
 """

# let's explore if the bins have a linear relationship
# with the target:

pd.concat([train_t, y_train], axis=1).groupby('age')['survived'].mean().plot()
plt.ylabel('mean of survived')
plt.show()

pd.concat([train_t, y_train], axis=1).groupby('fare')['survived'].mean().plot()
plt.ylabel('mean of survived')
plt.show()

# None of the variables show a monotonic relationship between the intervals of the discrete variable and the 
#       mean of survival. We can encode the intervals to return a monotonic relationship:

""" Ordinal encoding """

# perform ordinal encoding on train_t

# plot 

# Now we obtained a monotonic relationship between variables and target.

# That is all for this EXAMPLE. I hope you enjoyed the information, and see you in the next one.