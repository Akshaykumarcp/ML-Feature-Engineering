""" 
Target guided encodings
- In the previous examples in this section, we learned how to convert a label into a number, by using one hot encoding, 
    replacing by a digit or replacing by frequency or counts of observations. 
- These methods are simple, make (almost) no assumptions and work generally well in different scenarios.

There are however methods that allow us to capture information while pre-processing the labels of categorical variables. 

These methods include:

target refers to target feature/dependent feature

- Ordering mean and integer encoding labels according to the target
- Replacing labels by the target mean (mean encoding / target encoding)
- Replacing the labels by the probability ratio of the target being 1 or 0
- Weight of evidence.

All of the above methods have something in common:
- the encoding is guided by the target, and
- they create a monotonic relationship between the variable and the target.

Monotonicity
- A monotonic relationship is a relationship that does one of the following:
(1) as the value of one variable increases, so does the value of the other variable; or
(2) as the value of one variable increases, the value of the other variable decreases.

In this case, as the value of the independent variable (predictor) increases, so does the target, or conversely, as the value of the variable increases, the target value decreases.

Advantages of target guided encodings
- Capture information within the category, therefore creating more predictive features
- Create a monotonic relationship between the variable and the target, therefore suitable for linear models
- Do not expand the feature space

Limitations
- Prone to cause over-fitting
- Difficult to cross-validate with current libraries

Note
- The methods discussed in this and the coming 3 examples can be also used on numerical variables, after discretisation. 
- This creates a monotonic relationship between the numerical variable and the target, and therefore improves the performance of linear models. 

I will discuss this in more detail in the section "Discretisation".

===============================================================================

Weight of evidence
- Weight of Evidence (WoE) was developed primarily for the credit and financial industries to help build more predictive 
    models to evaluate the risk of loan default. 
    That is, to predict how likely the money lent to a person or institution is to be lost. 
    Thus, Weight of Evidence is a measure of the "strength” of a grouping technique to separate good and bad risk (default).

- WoE will be 0 if the P(Goods) / P(Bads) = 1, that is, if the outcome is random for that group.
- If P(Bads) > P(Goods) the odds ratio will be < 1 and,
- WoE will be < 0 if, P(Goods) > P(Bads).
- WoE is well suited for Logistic Regression, because the Logit transformation is simply the log of the odds, i.e., ln(P(Goods)/P(Bads)). Therefore, by using WoE-coded predictors in logistic regression, the predictors are all prepared and coded to the same scale, and the parameters in the linear logistic regression equation can be directly compared.

The WoE transformation has three advantages:
- It creates a monotonic relationship between the target and the independent variables.
- It orders the categories on a "logistic" scale which is natural for logistic regression
- The transformed variables can then be compared because they are on the same scale. Therefore, it is possible to determine which one is more predictive.

The WoE also has a limitation:
- Prone to cause over-fitting
- For more details follow this link

In this example:
We will see how to perform one hot encoding with:
- pandas

And the advantages and limitations of each implementation using the Titanic dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# load dataset
data = pd.read_csv('dataset/titanic.csv',
    usecols=['cabin', 'sex', 'embarked', 'survived'])

data.head()
"""  survived     sex cabin embarked
0         1  female    B5        S
1         1    male   C22        S
2         0  female   C22        S
3         0    male   C22        S
4         0  female   C22        S """

# let's remove observations with na in embarked
data.dropna(subset=['embarked'], inplace=True)

data.shape
# (1307, 4)

# Now we extract the first letter of the cabin
# to create a simpler variable for the demo

data['cabin'] = data['cabin'].astype(str).str[0]
# and we remove the observations where cabin = T
# because only few observations show T in the dataset

data = data[data['cabin']!= 'T']
data.shape
# (1306, 4)

# let's have a look at how many labels each variable has
for col in data.columns:
    print(col, ': ', len(data[col].unique()), ' labels')

""" survived :  2  labels
sex :  2  labels
cabin :  8  labels
embarked :  3  labels """

# let's explore the unique categories
data['cabin'].unique()
# array(['B', 'C', 'E', 'D', 'A', 'n', 'F', 'G'], dtype=object)

data['sex'].unique()
# array(['female', 'male'], dtype=object)

data['embarked'].unique()
# array(['S', 'C', 'Q'], dtype=object)

"""
Encoding important
- We select calculate the woe using the train set, and then use those mappings in the test set.

Note that in the pandas implementation, we need to keep the target in the training set
 """

# let's separate into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    data[['cabin', 'sex', 'embarked', 'survived']],  # this time we keep the target!!
    data['survived'],  # target
    test_size=0.3,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape
# ((914, 4), (392, 4))

# Explore original relationship between categorical variables and target
# let's explore the relationship of the categories with the target
for var in ['cabin', 'sex', 'embarked']:
    fig = plt.figure()
    fig = X_train.groupby([var])['survived'].mean().plot()
    fig.set_title('Relationship between {} and Survival'.format(var))
    fig.set_ylabel('Mean Survival')
    plt.show()

""" 
OBSERVATION:
- You can see that the relationship between the target and cabin and embarked goes up and down, depending on the category.

Weight of Evidence with pandas

Advantages
- quick
- returns pandas dataframe

Limitations of pandas:
- it does not preserve information from train data to propagate to test data
- We need to store the mappings separately if planning to use the model in production. """

# let's calculate the probability of survived = 1 per category
# (probability of events or p(1))

prob_df = X_train.groupby(['cabin'])['survived'].mean()

# and capture it into a dataframe
prob_df = pd.DataFrame(prob_df)

prob_df
""" survived cabin	
A	0.411765
B	0.738095
C	0.600000
D	0.696970
E	0.700000
F	0.769231
G	0.750000
n	0.292199 """

# and now the probability of survived = 0
# (probability of non-events or p(0))

prob_df['died'] = 1-prob_df['survived']

prob_df
""" survived	died cabin		
A	0.411765	0.588235
B	0.738095	0.261905
C	0.600000	0.400000
D	0.696970	0.303030
E	0.700000	0.300000
F	0.769231	0.230769
G	0.750000	0.250000
n	0.292199	0.707801 """

#  and now the woe
prob_df['ratio'] = np.log( prob_df['survived'] / prob_df['died'] )

prob_df
""" survived	died	ratio   cabin			
A	0.411765	0.588235	-0.356675
B	0.738095	0.261905	1.036092
C	0.600000	0.400000	0.405465
D	0.696970	0.303030	0.832909
E	0.700000	0.300000	0.847298
F	0.769231	0.230769	1.203973
G	0.750000	0.250000	1.098612
n	0.292199	0.707801	-0.884730 """

# and now let's capture the woe in a dictionary
ordered_labels = prob_df['ratio'].to_dict()

ordered_labels
""" {'A': -0.35667494393873245,
 'B': 1.0360919316867758,
 'C': 0.4054651081081642,
 'D': 0.8329091229351043,
 'E': 0.8472978603872034,
 'F': 1.2039728043259361,
 'G': 1.0986122886681098,
 'n': -0.8847299269619376} """

# now, we replace the labels with the woe
X_train['cabin'] = X_train['cabin'].map(ordered_labels)
X_test['cabin'] = X_test['cabin'].map(ordered_labels)

# let's explore the result
X_train['cabin'].head(10)
"""
843    -0.884730
869    -0.884730
430    -0.884730
481    -0.884730
1308   -0.884730
456    -0.884730
118     0.832909
485     0.832909
297    -0.884730
263     0.847298
Name: cabin, dtype: float64 """

# we can turn the previous commands into 2 functions
def find_category_mappings(df, variable, target):
    tmp = pd.DataFrame(df.groupby([variable])[target].mean())
    tmp['non-target'] = 1 - tmp[target]
    tmp['ratio'] = np.log( tmp[target] / tmp['non-target'] )
    return tmp['ratio'].to_dict()

def integer_encode(train, test, variable, ordinal_mapping):
    X_train[variable] = X_train[variable].map(ordinal_mapping)
    X_test[variable] = X_test[variable].map(ordinal_mapping)

# and now we run a loop over the remaining categorical variables
for variable in ['sex', 'embarked']:
    mappings = find_category_mappings(X_train, variable, 'survived')
    integer_encode(X_train, X_test, variable, mappings)

# let's see the result
X_train.head()
"""     cabin       sex  embarked  survived
843  -0.88473 -1.522770 -0.674455         0
869  -0.88473 -1.522770 -0.674455         0
430  -0.88473  1.025409 -0.674455         1
481  -0.88473  1.025409  0.149212         1
1308 -0.88473 -1.522770 -0.674455         0 """

# let's inspect the newly created monotonic relationship
# between the categorical variables and the target
for var in ['cabin', 'sex', 'embarked']:
    fig = plt.figure()
    fig = X_train.groupby([var])['survived'].mean().plot()
    fig.set_title('Monotonic relationship between {} and Survival'.format(var))
    fig.set_ylabel('Mean Survived')
    plt.show()

"""
OBSERVATION:
- Note the monotonic relationships between the mean target and the categories.

Note
- Replacing categorical labels with this code and method will generate missing values for categories present in the test set that were not seen in the training set. Therefore it is extremely important to handle rare labels before-hand. I will explain how to do this, in a later notebook.

In addition, it will create NA or Inf if the probability of target = 0 is zero, as the division by zero is not defined. """