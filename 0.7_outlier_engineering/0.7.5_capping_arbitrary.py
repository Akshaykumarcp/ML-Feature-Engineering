""" 
Outlier Engineering
- An outlier is a data point which is significantly different from the remaining data. 
- “An outlier is an observation which deviates so much from the other observations as to arouse suspicions that it 
        was generated by a different mechanism.” [D. Hawkins. Identification of Outliers, Chapman and Hall , 1980].

- Statistics such as the mean and variance are very susceptible to outliers. 
- In addition, some Machine Learning models are sensitive to outliers which may decrease their performance. 
- Thus, depending on which algorithm we wish to train, we often remove outliers from our variables.

- In section 0.2.5 of this series we have seen how to identify outliers. 
- In this example, we we discuss how we can process them to train our machine learning models.

How can we pre-process outliers?
- Trimming: remove the outliers from our dataset
- Treat outliers as missing data, and proceed with any missing data imputation technique
- Discrestisation: outliers are placed in border bins together with higher or lower values of the distribution
- Censoring: capping the variable distribution at a max and / or minimum value
        Censoring is also known as:
        - top and bottom coding
        - winsorization
        - capping

Censoring or Capping.
- Censoring, or capping, means capping the maximum and /or minimum of a distribution at an arbitrary value.
- On other words, values bigger or smaller than the arbitrarily determined ones are censored.

- Capping can be done at both tails, or just one of the tails, depending on the variable and the user.

- Check my talk in pydata for an example of capping used in a finance company.

- The numbers at which to cap the distribution can be determined:
        - arbitrarily
        - using the inter-quantal range proximity rule
        - using the gaussian approximation
        - using quantiles

Advantages
- does not remove data

Limitations
- distorts the distributions of the variables
- distorts the relationships among variables

In this example
- We will see how to perform capping with arbitrary values using the Titanic dataset

Important
- Outliers should be detected AND removed ONLY from the training set, and NOT from the test set.
- So we should first divide our data set into train and tests, and remove outliers in the train set, 
        but keep those in the test set, and measure how well our model is doing. 
        
I will not do that in this example, but please keep that in mind when setting up your pipelines """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_engine import outliers as outr

def load_titanic():
    data = pd.read_csv('dataset/titanic.csv')
    data['cabin'] = data['cabin'].astype(str).str[0]
    data['pclass'] = data['pclass'].astype('O')
    data['embarked'].fillna('C', inplace=True)
    return data

data = load_titanic()

data.head()
""" 
  pclass  survived                                             name     sex      age  sibsp  parch  ticket      fare cabin embarked boat   body                        home.dest
0      1         1                    Allen, Miss. Elisabeth Walton  female  29.0000      0      0   24160  211.3375     B        S    2    NaN                     St Louis, MO
1      1         1                   Allison, Master. Hudson Trevor    male   0.9167      1      2  113781  151.5500     C        S   11    NaN  Montreal, PQ / Chesterville, ON
2      1         0                     Allison, Miss. Helen Loraine  female   2.0000      1      2  113781  151.5500     C        S  NaN    NaN  Montreal, PQ / Chesterville, ON
3      1         0             Allison, Mr. Hudson Joshua Creighton    male  30.0000      1      2  113781  151.5500     C        S  NaN  135.0  Montreal, PQ / Chesterville, ON
4      1         0  Allison, Mrs. Hudson J C (Bessie Waldo Daniels)  female  25.0000      1      2  113781  151.5500     C        S  NaN    NaN  Montreal, PQ / Chesterville, ON

ArbitraryOutlierCapper
- The ArbitraryOutlierCapper caps the minimum and maximum values by a value determined by the user.
 """

data.shape
# (1309, 14)

# let's find out the maximum Age and maximum Fare in the titanic
data.age.max(), data.fare.max()
# (80.0, 512.3292)

# check for missing values before doing ArbitraryOutlierCapper
data.isnull().sum()
""" 
pclass          0
survived        0
name            0
sex             0
age           263
sibsp           0
parch           0
ticket          0
fare            1
cabin           0
embarked        0
boat          823
body         1188
home.dest     564
dtype: int64 """

# missing values exists, to focus on ArbitraryOutlierCapper, lets drop NAN for 2 features
# Credits: https://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-in-a-certain-column-is-nan
data = data[data['age'].notnull()]
data = data[data['fare'].notnull()]

data.isnull().sum()
""" 
pclass         0
survived       0
name           0
sex            0
age            0
sibsp          0
parch          0
ticket         0
fare           0
cabin          0
embarked       0
boat         628
body         926
home.dest    360
dtype: int64 """

data.shape
# (1045, 14)

capper = outr.ArbitraryOutlierCapper(max_capping_dict = {'age':50, 'fare':200},
                                     min_capping_dict = None)

capper.fit(data)
# ArbitraryOutlierCapper(max_capping_dict={'age': 50, 'fare': 200})

capper.right_tail_caps_
# {'age': 50, 'fare': 200}

capper.left_tail_caps_
# {}

temp = capper.transform(data)

temp.age.max(), temp.fare.max()
# (50.0, 200.0)

# Minimum capping
capper = outr.ArbitraryOutlierCapper(max_capping_dict=None,
                                     min_capping_dict={
                                         'age': 10,
                                         'fare': 100
                                     })
capper.fit(data)
# ArbitraryOutlierCapper(min_capping_dict={'age': 10, 'fare': 100})

capper.variables
# ['age', 'fare']

capper.right_tail_caps_
# {}

capper.left_tail_caps_
# {'age': 10, 'fare': 100}

temp = capper.transform(data)

temp.age.min(), temp.fare.min()
# (10.0, 100.0)

# Both ends capping
capper = outr.ArbitraryOutlierCapper(max_capping_dict={
                                     'age': 50, 'fare': 200},
                                     min_capping_dict={
                                     'age': 10, 'fare': 100})
capper.fit(data)
# ArbitraryOutlierCapper(max_capping_dict={'age': 50, 'fare': 200},
#                       min_capping_dict={'age': 10, 'fare': 100})

capper.right_tail_caps_
# {'age': 50, 'fare': 200}

capper.left_tail_caps_
# {'age': 10, 'fare': 100}

temp = capper.transform(data)

temp.age.min(), temp.fare.min()
# (10.0, 100.0)

temp.age.max(), temp.fare.max()
# (50.0, 200.0)

# That is all for this example. I hope you enjoyed the information, and see you in the next one. 