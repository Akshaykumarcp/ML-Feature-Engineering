""" 
Domain knowledge discretisation
- Frequently, when engineering variables in a business setting, the business experts determine the intervals in 
        which they think the variable should be divided so that it makes sense for the business. 
        Typical examples are the discretisation of variables like Age and Income.

- Income for example is usually capped at a certain maximum value, and all incomes above that value fall into the 
        last bucket. As per Age, it is usually divided in certain groups according to the business need, 
        for example division into 0-21 (for under-aged), 20-30 (for young adults), 30-40, 40-60, and > 60 
        (for retired or close to) are frequent.

In this example
- We will learn how to divide a variable into pre-defined buckets using the titanic and lending club datasets.
 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# load the Titanic Dataset

data = pd.read_csv('dataset/titanic.csv',usecols=['age', 'survived'])

data.head()
"""    
    survived      age
0         1  29.0000
1         1   0.9167
2         0   2.0000
3         0  30.0000
4         0  25.0000 """

# The variable Age contains missing data, that I will fill by extracting a random sample of the variable.

def impute_na(data, variable):
    df = data.copy()

    # random sampling
    df[variable+'_random'] = df[variable]

    # extract the random sample to fill the na
    random_sample = data[variable].dropna().sample(
        df[variable].isnull().sum(), random_state=0)

    # pandas needs to have the same index in order to merge datasets
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(), variable+'_random'] = random_sample

    return df[variable+'_random']

# let's fill the missing data
data['age'] = impute_na(data, 'age')

# let's divide Age into the buckets

# bucket boundaries
buckets = [0, 20, 40, 60, 1000]

# bucket labels
labels = ['0-20', '20-40', '40-60', '>60']

# discretisation
data['Age_buckets_labels'] = pd.cut(data['age'], bins=buckets, labels=labels, include_lowest=True)

data['Age_buckets'] = pd.cut(data['age'], bins=buckets, include_lowest=True)

data.head()
""" 
   survived      age Age_buckets_labels     Age_buckets
0         1  29.0000              20-40    (20.0, 40.0]
1         1   0.9167               0-20  (-0.001, 20.0]
2         0   2.0000               0-20  (-0.001, 20.0]
3         0  30.0000              20-40    (20.0, 40.0]
4         0  25.0000              20-40    (20.0, 40.0] """

data.tail()
""" 
      survived   age Age_buckets_labels     Age_buckets
1304         0  14.5               0-20  (-0.001, 20.0]
1305         0  39.0              20-40    (20.0, 40.0]
1306         0  26.5              20-40    (20.0, 40.0]
1307         0  27.0              20-40    (20.0, 40.0]
1308         0  29.0              20-40    (20.0, 40.0] 

- Above we can observe the buckets into which each Age observation was placed. 
- For example, age 14 was placed into the 0-20 bucket.

- Let's explore the number of observations and survival rate per bucket after this arbitrary discretisation method.
"""

# number of passengers per age bucket

data.groupby('Age_buckets_labels')['age'].count().plot.bar()
plt.show()

# survival rate per age bucket

data.groupby('Age_buckets_labels')['survived'].mean().plot.bar()
plt.show()

""" 
Lending Club
- Let's explore discretisation using domain knowledge in a different business scenario. 
- I will use the loan book from the peer to peer lending company Lending Club. 
- This dataset contains information on loans given to people, and the financial characteristics of those
         people as well as the loan performance.
 """

# I will load only the income declared by the borrower for the demonstration
data = pd.read_csv('dataset/loan.csv/loan.csv', usecols=['annual_inc'])
data.head()
""" 
   annual_inc
0     24000.0
1     30000.0
2     12252.0
3     49200.0
4     80000.0 """

data.annual_inc.describe()
""" 
count    8.873750e+05
mean     7.502759e+04
std      6.469830e+04
min      0.000000e+00
25%      4.500000e+04
50%      6.500000e+04
75%      9.000000e+04
max      9.500000e+06
Name: annual_inc, dtype: float64 """

# let's inspect the distribution of Incomes

data.annual_inc.hist(bins=100)
plt.show()

# and now let's look at the lower incomes in more detail

data[data.annual_inc<500000].annual_inc.hist(bins=100)
plt.show()

# We can see that the majority of the population earns below 150,000. So we may want to make a cap there.

# and now let's divide into arbitrary buckets, assuming that these make business sense

# bucket interval
buckets = [0, 45000, 65000, 90000, 150000, 1e10]

# bucket labels
labels = ['0-45k', '45-65k', '65-90k', '90-150k', '>150k']

# discretisation
data['Income_buckets'] = pd.cut(
    data.annual_inc, bins=buckets, labels=labels, include_lowest=True)

data.head()
"""    annual_inc Income_buckets
0     24000.0          0-45k
1     30000.0          0-45k
2     12252.0          0-45k
3     49200.0         45-65k
4     80000.0         65-90k """

data.tail()
"""
         annual_inc Income_buckets
887374     31000.0          0-45k
887375     79000.0         65-90k
887376     35000.0          0-45k
887377     64400.0         45-65k
887378    100000.0        90-150k """

data.groupby(['Income_buckets'])['annual_inc'].count().plot.bar()
plt.xticks(rotation=45)
plt.show()

(data.groupby(['Income_buckets'])['annual_inc'].count()/len(data)).plot.bar()
plt.xticks(rotation=45)
plt.show()

""" 
- We have captured ~equal amount of borrowers in each of the first 3 buckets, and we see clearly, 
        that a smaller percentage of the loans were disbursed to high earners.

That is all for this example. I hope you enjoyed the info, and see you in the next one. """