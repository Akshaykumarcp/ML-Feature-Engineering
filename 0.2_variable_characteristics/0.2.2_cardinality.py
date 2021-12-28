""" 
Cardinality
- The values of a categorical variable are selected from a group of categories, also called labels. 
- For example, in the variable gender the categories or labels are male and female, 
    whereas in the variable city the labels can be London, Manchester, Brighton and so on.

Different categorical variables contain different number of labels or categories. 
    The variable gender contains only 2 labels, 
    but a variable like city or postcode, can contain a huge number of different labels.

The number of different labels within a categorical variable is known as cardinality. 
    A high number of labels within a variable is known as high cardinality.

Are multiple labels in a categorical variable a problem?
High cardinality may pose the following problems:
- Variables with too many labels tend to dominate over those with only a few labels, 
    particularly in Tree based algorithms.

- A big number of labels within a variable may introduce noise with little, 
    if any, information, therefore making machine learning models prone to over-fit.

- Some of the labels may only be present in the training data set, but not in the test set, 
    therefore machine learning algorithms may over-fit to the training set.

- Contrarily, some labels may appear only in the test set, therefore leaving the machine learning 
    algorithms unable to perform a calculation over the new (unseen) observation.

- In particular, tree methods can be biased towards variables with lots of labels (variables with high cardinality). 
    Thus, their performance may be affected by high cardinality.

Below, I will show the effect of high cardinality of variables on the performance of different machine learning 
algorithms, and how a quick fix to reduce the number of labels, without any sort of data insight, 
already helps to boost performance.

In this example:
We will:
- Learn how to quantify cardinality
- See examples of high and low cardinality variables
- Understand the effect of cardinality when preparing train and test sets
- Visualise the effect of cardinality on Machine Learning Model performance

We will use the Titanic dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# let's load the titanic dataset
data = pd.read_csv('dataset/titanic.csv')

data.head()
""" 
   pclass  survived                                             name     sex  \
0       1         1                    Allen, Miss. Elisabeth Walton  female
1       1         1                   Allison, Master. Hudson Trevor    male
2       1         0                     Allison, Miss. Helen Loraine  female
3       1         0             Allison, Mr. Hudson Joshua Creighton    male
4       1         0  Allison, Mrs. Hudson J C (Bessie Waldo Daniels)  female

       age  sibsp  parch  ticket      fare cabin embarked boat   body  \
0  29.0000      0      0   24160  211.3375    B5        S    2    NaN
1   0.9167      1      2  113781  151.5500   C22        S   11    NaN
2   2.0000      1      2  113781  151.5500   C22        S  NaN    NaN
3  30.0000      1      2  113781  151.5500   C22        S  NaN  135.0
4  25.0000      1      2  113781  151.5500   C22        S  NaN    NaN

                         home.dest
0                     St Louis, MO
1  Montreal, PQ / Chesterville, ON
2  Montreal, PQ / Chesterville, ON
3  Montreal, PQ / Chesterville, ON
4  Montreal, PQ / Chesterville, ON """

""" 
The categorical variables in this dataset are Name, Sex, Ticket, Cabin and Embarked.

Note that Ticket and Cabin contain both letters and numbers, so they could be treated as Mixed Variables. 
For this example, I will treat them as categorical.
 """

# let's inspect the cardinality, this is the number
# of different labels, for the different categorical variables

print('Number of categories in the variable Name: {}'.format(len(data.name.unique())))

print('Number of categories in the variable Gender: {}'.format(len(data.sex.unique())))

print('Number of categories in the variable Ticket: {}'.format(len(data.ticket.unique())))

print('Number of categories in the variable Cabin: {}'.format(len(data.cabin.unique())))

print('Number of categories in the variable Embarked: {}'.format(len(data.embarked.unique())))

""" 
Number of categories in the variable Name: 1307
Number of categories in the variable Gender: 2
Number of categories in the variable Ticket: 929
Number of categories in the variable Cabin: 182
Number of categories in the variable Embarked: 4 """


print('Total number of passengers in the Titanic: {}'.format(len(data)))
# Total number of passengers in the Titanic: 1309

""" 
- While the variable Sex contains only 2 categories and 
- Embarked 4 (low cardinality), 
- the variables Ticket, Name and Cabin, as expected, contain a huge number of different labels (high cardinality).

To demonstrate the effect of high cardinality in train and test sets and machine learning performance, 
- I will work with the variable Cabin. 
- I will create a new variable with reduced cardinality.
"""

# let's explore the values / categories of Cabin

# we know from the previous outcome that there are 148
# different cabins, therefore the variable
# is highly cardinal

data.cabin.unique()
""" array(['B5', 'C22', 'E12', 'D7', 'A36', 'C101', nan, 'C62', 'B35', 'A23',
       'B58', 'D15', 'C6', 'D35', 'C148', 'C97', 'B49', 'C99', 'C52', 'T',
       'A31', 'C7', 'C103', 'D22', 'E33', 'A21', 'B10', 'B4', 'E40',
       'B38', 'E24', 'B51', 'B96', 'C46', 'E31', 'E8', 'B61', 'B77', 'A9',
       'C89', 'A14', 'E58', 'E49', 'E52', 'E45', 'B22', 'B26', 'C85',
       'E17', 'B71', 'B20', 'A34', 'C86', 'A16', 'A20', 'A18', 'C54',
       'C45', 'D20', 'A29', 'C95', 'E25', 'C111', 'C23', 'E36', 'D34',
       'D40', 'B39', 'B41', 'B102', 'C123', 'E63', 'C130', 'B86', 'C92',
       'A5', 'C51', 'B42', 'C91', 'C125', 'D10', 'B82', 'E50', 'D33',
       'C83', 'B94', 'D49', 'D45', 'B69', 'B11', 'E46', 'C39', 'B18',
       'D11', 'C93', 'B28', 'C49', 'B52', 'E60', 'C132', 'B37', 'D21',
       'D19', 'C124', 'D17', 'B101', 'D28', 'D6', 'D9', 'B80', 'C106',
       'B79', 'C47', 'D30', 'C90', 'E38', 'C78', 'C30', 'C118', 'D36',
       'D48', 'D47', 'C105', 'B36', 'B30', 'D43', 'B24', 'C2', 'C65',
       'B73', 'C104', 'C110', 'C50', 'B3', 'A24', 'A32', 'A11', 'A10',
       'B57', 'C28', 'E44', 'A26', 'A6', 'A7', 'C31', 'A19', 'B45', 'E34',
       'B78', 'B50', 'C87', 'C116', 'C55', 'D50', 'E68', 'E67', 'C126',
       'C68', 'C70', 'C53', 'B19', 'D46', 'D37', 'D26', 'C32', 'C80',
       'C82', 'C128', 'E39', 'D', 'F4', 'D56', 'F33', 'E101', 'E77', 'F2',
       'D38', 'F', 'E121', 'E10', 'G6', 'F38'], dtype=object) """
  
"""      
- Let's now reduce the cardinality of the variable. 

How? 
- instead of using the entire cabin value, I will capture only the first letter.
- Rationale: the first letter indicates the deck on which the cabin was located, and is therefore an indication of 
    both social class status and proximity to the surface of the Titanic. 
    Both are known to improve the probability of survival.
 """
# let's capture the first letter of Cabin
data['Cabin_reduced'] = data['cabin'].astype(str).str[0]

data[['cabin', 'Cabin_reduced']].head()
""" cabin	Cabin_reduced
0	B5	B
1	C22	C
2	C22	C
3	C22	C
4	C22	C """

print('Number of categories in the variable Cabin: {}'.format(len(data.cabin.unique())))

print('Number of categories in the variable Cabin reduced: {}'.format(len(data.Cabin_reduced.unique())))

""" 
Number of categories in the variable Cabin: 182
Number of categories in the variable Cabin reduced: 9

- We reduced the number of different labels from 182 to 9.
 """

# let's separate into training and testing set
# in order to build machine learning models

use_cols = ['cabin', 'Cabin_reduced', 'sex']

# this functions comes from scikit-learn
X_train, X_test, y_train, y_test = train_test_split(
    data[use_cols], 
    data['survived'],  
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape
# ((916, 3), (393, 3))

""" 
High cardinality leads to uneven distribution of categories in train and test sets
When a variable is highly cardinal, often some categories land only on the training set, or only on the testing set. 
- If present only in the training set, they may lead to over-fitting. 
- If present only on the testing set, the machine learning algorithm will not know how to handle them, 
    as it has not seen them during training. """

# Let's find out labels present only in the training set

unique_to_train_set = [x for x in X_train.cabin.unique() if x not in X_test.cabin.unique()]

len(unique_to_train_set)
# 113

# There are 113 Cabins only present in the training set, and not in the testing set.

# Let's find out labels present only in the test set

unique_to_test_set = [x for x in X_test.cabin.unique() if x not in X_train.cabin.unique()]

len(unique_to_test_set)
# 36

""" 
- Variables with high cardinality tend to have values (i.e., categories) present in the training set,
    that are not present in the test set, and vice versa. 
- This will bring problems at the time of training (due to over-fitting) and scoring of new data 
    (how should the model deal with unseen categories?).

This problem is almost overcome by reducing the cardinality of the variable. See below.
 """
# Let's find out labels present only in the training set
# for Cabin with reduced cardinality

unique_to_train_set = [
    x for x in X_train['Cabin_reduced'].unique()
    if x not in X_test['Cabin_reduced'].unique()
]

len(unique_to_train_set)
# 1

# Let's find out labels present only in the test set
# for Cabin with reduced cardinality

unique_to_test_set = [
    x for x in X_test['Cabin_reduced'].unique()
    if x not in X_train['Cabin_reduced'].unique()
]

len(unique_to_test_set)
# 0
""" 
- Observe how by reducing the cardinality there is now only 1 label in the training set that is not present in the test set.
- And no label in the test set that is not contained in the training set as well.

Effect of cardinality on Machine Learning Model Performance
- In order to evaluate the effect of categorical variables in machine learning models,
- I will quickly replace the categories by numbers. See below.
 """
# Let's re-map Cabin into numbers so we can use it to train ML models

# I will replace each cabin by a number
# to quickly demonstrate the effect of
# labels on machine learning algorithms

##############
# Note: this is neither the only nor the best
# way to encode categorical variables into numbers
# there is more on these techniques in the section
# "Encoding categorical variales"
##############

cabin_dict = {k: i for i, k in enumerate(X_train.cabin.unique(), 0)}
cabin_dict
""" {nan: 0,
 'E36': 1,
 'C68': 2,
 'E24': 3,
 'C22': 4,
 'D38': 5,
 'B50': 6,
 'A24': 7,
 'C111': 8,
 'F': 9,
 'C6': 10,
 'C87': 11,
 'E8': 12,
 'B45': 13,
 'C93': 14,
 'D28': 15,
 'D36': 16,
 'C125': 17,
 'B35': 18,
 'T': 19,
 'B73': 20,
 'B57': 21,
 'A26': 22,
 'A18': 23,
 'B96': 24,
 'G6': 25,
 'C78': 26,
 'C101': 27,
 'D9': 28,
 'D33': 29,
 'C128': 30,
 'E50': 31,
 'B26': 32,
 'B69': 33,
 'E121': 34,
 'C123': 35,
 'B94': 36,
 'A34': 37,
 'D': 38,
 'C39': 39,
 'D43': 40,
 'E31': 41,
 'B5': 42,
 'D17': 43,
 'F33': 44,
 'E44': 45,
 'D7': 46,
 'A21': 47,
 'D34': 48,
 'A29': 49,
 'D35': 50,
 'A11': 51,
 'B51': 52,
 'D46': 53,
 'E60': 54,
 'C30': 55,
 'D26': 56,
 'E68': 57,
 'A9': 58,
 'B71': 59,
 'D37': 60,
 'F2': 61,
 'C55': 62,
 'C89': 63,
 'C124': 64,
 'C23': 65,
 'C126': 66,
 'E49': 67,
 'E46': 68,
 'D19': 69,
 'B58': 70,
 'C82': 71,
 'B52': 72,
 'C92': 73,
 'E45': 74,
 'C65': 75,
 'E25': 76,
 'B3': 77,
 'D40': 78,
 'C91': 79,
 'B102': 80,
 'B61': 81,
 'A20': 82,
 'B36': 83,
 'C7': 84,
 'B77': 85,
 'D20': 86,
 'C148': 87,
 'C105': 88,
 'E38': 89,
 'B86': 90,
 'C132': 91,
 'C86': 92,
 'A14': 93,
 'C54': 94,
 'A5': 95,
 'B49': 96,
 'B28': 97,
 'B24': 98,
 'C2': 99,
 'F4': 100,
 'A6': 101,
 'C83': 102,
 'B42': 103,
 'A36': 104,
 'C52': 105,
 'D56': 106,
 'C116': 107,
 'B19': 108,
 'E77': 109,
 'E101': 110,
 'B18': 111,
 'C95': 112,
 'D15': 113,
 'E33': 114,
 'B30': 115,
 'D21': 116,
 'E10': 117,
 'C130': 118,
 'D6': 119,
 'C51': 120,
 'D30': 121,
 'E67': 122,
 'C110': 123,
 'C103': 124,
 'C90': 125,
 'C118': 126,
 'C97': 127,
 'D47': 128,
 'E34': 129,
 'B4': 130,
 'D50': 131,
 'C62': 132,
 'E17': 133,
 'B41': 134,
 'C49': 135,
 'C85': 136,
 'B20': 137,
 'C28': 138,
 'E63': 139,
 'C99': 140,
 'D49': 141,
 'A10': 142,
 'A16': 143,
 'B37': 144,
 'C80': 145,
 'B78': 146}

 """

# replace the labels in Cabin, using the dic created above
X_train.loc[:, 'Cabin_mapped'] = X_train.loc[:, 'cabin'].map(cabin_dict)
X_test.loc[:, 'Cabin_mapped'] = X_test.loc[:, 'cabin'].map(cabin_dict)

X_train[['Cabin_mapped', 'cabin']].head(10)
"""
Cabin_mapped	cabin
501	0	NaN
588	0	NaN
402	0	NaN
1193	0	NaN
686	0	NaN
971	0	NaN
117	1	E36
540	0	NaN
294	2	C68
261	3	E24 """

# We see how NaN takes the value 0 in the new variable, E36 takes the value 1, C68 takes the value 2, and so on.
# Now I will replace the letters in the reduced cabin variable with the same procedure

# create replace dictionary
cabin_dict = {k: i for i, k in enumerate(X_train['Cabin_reduced'].unique(), 0)}

# replace labels by numbers with dictionary
X_train.loc[:, 'Cabin_reduced'] = X_train.loc[:, 'Cabin_reduced'].map(cabin_dict)
X_test.loc[:, 'Cabin_reduced'] = X_test.loc[:, 'Cabin_reduced'].map(cabin_dict)

X_train[['Cabin_reduced', 'cabin']].head(20)
""" 
    Cabin_reduced	cabin
501	0	NaN
588	0	NaN
402	0	NaN
1193	0	NaN
686	0	NaN
971	0	NaN
117	1	E36
540	0	NaN
294	2	C68
261	1	E24
587	0	NaN
489	0	NaN
2	2	C22
405	0	NaN
1284	0	NaN
338	0	NaN
356	0	NaN
985	0	NaN
182	0	NaN
1027	0	NaN """

# We see now that E36 and E24 take the same number, 1, because we are capturing only the letter. 
# They both start with E.

# re-map the categorical variable Sex into numbers

X_train.loc[:, 'sex'] = X_train.loc[:, 'sex'].map({'male': 0, 'female': 1})
X_test.loc[:, 'sex'] = X_test.loc[:, 'sex'].map({'male': 0, 'female': 1})

X_train.sex.head()
"""
501     1
588     1
402     1
1193    0
686     1
Name: sex, dtype: int64 """

# check if there are missing values in these variables

X_train[['Cabin_mapped', 'Cabin_reduced', 'sex']].isnull().sum()
""" 
Cabin_mapped     0
Cabin_reduced    0
sex              0
dtype: int64 """

X_test[['Cabin_mapped', 'Cabin_reduced', 'sex']].isnull().sum()
""" 
Cabin_mapped     41
Cabin_reduced     0
sex               0
dtype: int64 """

# In the test set, there are now 30 missing values for the highly cardinal variable. 
# These were introduced when encoding the categories into numbers.
#  How? 
# Many categories exist only in the test set. Thus, when we created our encoding dictionary using only the train set, 
#   we did not generate a number to replace those labels present only in the test set. 
#   As a consequence, they were encoded as NaN. We will see in future examples how to tackle this problem. 
#   For now, I will fill those missing values with -1.

# let's check the number of different categories in the encoded variables
len(X_train.Cabin_mapped.unique()), len(X_train.Cabin_reduced.unique())
# (147, 9)

""" From the above we note immediately that from the original 148 cabins in the dataset, 
    only 121 are present in the training set. We also see how we reduced the number of different categories to 
    just 9 in our previous step.

Let's go ahead and evaluate the effect of labels in machine learning algorithms.
 """

# Random Forests
# model built on data with high cardinality for cabin

# call the model
rf = RandomForestClassifier(n_estimators=200, random_state=39)

# train the model
rf.fit(X_train[['Cabin_mapped', 'sex']], y_train)

# make predictions on train and test set
pred_train = rf.predict_proba(X_train[['Cabin_mapped', 'sex']])
pred_test = rf.predict_proba(X_test[['Cabin_mapped', 'sex']].fillna(0))

print('Train set')
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))

""" 
Train set
Random Forests roc-auc: 0.853790650048556
Test set
Random Forests roc-auc: 0.7691361097284443

- We observe that the performance of the Random Forests on the training set is quite superior to its performance in
 the test set. This indicates that the model is over-fitting, which means that it does a great job at predicting 
 the outcome on the dataset it was trained on, but it lacks the power to generalise the prediction to unseen data.
 """
# model built on data with low cardinality for cabin

# call the model
rf = RandomForestClassifier(n_estimators=200, random_state=39)

# train the model
rf.fit(X_train[['Cabin_reduced', 'sex']], y_train)

# make predictions on train and test set
pred_train = rf.predict_proba(X_train[['Cabin_reduced', 'sex']])
pred_test = rf.predict_proba(X_test[['Cabin_reduced', 'sex']])

print('Train set')
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))

""" 
Train set
Random Forests roc-auc: 0.8163420365403872
Test set
Random Forests roc-auc: 0.8017670482827277

- We can see now that the Random Forests no longer over-fit to the training set.
-  In addition, the model is much better at generalising the predictions 
        (compare the roc-auc of this model on the test set vs the roc-auc of the model above also in the test set: 0.81 vs 0.80).

I would like to point out, that likely we can overcome the effect of high cardinality by adjusting the hyper-parameters 
of the random forests. That goes beyond the scope. Here, I want to show you that given a same model, 
with identical hyper-parameters, high cardinality may cause the model to over-fit.
 """

# AdaBoost
# model build on data with plenty of categories in Cabin

# call the model
ada = AdaBoostClassifier(n_estimators=200, random_state=44)

# train the model
ada.fit(X_train[['Cabin_mapped', 'sex']], y_train)

# make predictions on train and test set
pred_train = ada.predict_proba(X_train[['Cabin_mapped', 'sex']])
pred_test = ada.predict_proba(X_test[['Cabin_mapped', 'sex']].fillna(0))

print('Train set')
print('Adaboost roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Adaboost roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))

""" 
Train set
Adaboost roc-auc: 0.8296861713101102
Test set
Adaboost roc-auc: 0.7604391350035948 """

# model build on data with fewer categories in Cabin Variable

# call the model
ada = AdaBoostClassifier(n_estimators=200, random_state=44)

# train the model
ada.fit(X_train[['Cabin_reduced', 'sex']], y_train)

# make predictions on train and test set
pred_train = ada.predict_proba(X_train[['Cabin_reduced', 'sex']])
pred_test = ada.predict_proba(X_test[['Cabin_reduced', 'sex']].fillna(0))

print('Train set')
print('Adaboost roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Adaboost roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))

""" 
Train set
Adaboost roc-auc: 0.8161256723642566
Test set
Adaboost roc-auc: 0.8001078480172557

- Similarly, the Adaboost model trained on the variable with high cardinality is overfit to the train set. 
- Whereas the Adaboost trained on the low cardinal variable is not overfitting and therefore does a better job in
     generalising the predictions.

In addition, building an AdaBoost on a model with less categories in Cabin, is a) simpler and b) should a different category in the test set appear, by taking just the front letter of cabin, the ML model will know how to handle it because it was seen during training.
 """

# Logistic Regression
# model build on data with plenty of categories in Cabin variable

# call the model
logit = LogisticRegression(random_state=44, solver='lbfgs')

# train the model
logit.fit(X_train[['Cabin_mapped', 'sex']], y_train)

# make predictions on train and test set
pred_train = logit.predict_proba(X_train[['Cabin_mapped', 'sex']])
pred_test = logit.predict_proba(X_test[['Cabin_mapped', 'sex']].fillna(0))

print('Train set')
print('Logistic regression roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Logistic regression roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))

"""
Train set
Logistic regression roc-auc: 0.8133909298124677
Test set
Logistic regression roc-auc: 0.7750815773463858 """

# model build on data with fewer categories in Cabin Variable

# call the model
logit = LogisticRegression(random_state=44, solver='lbfgs')

# train the model
logit.fit(X_train[['Cabin_reduced', 'sex']], y_train)

# make predictions on train and test set
pred_train = logit.predict_proba(X_train[['Cabin_reduced', 'sex']])
pred_test = logit.predict_proba(X_test[['Cabin_reduced', 'sex']].fillna(0))

print('Train set')
print('Logistic regression roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Logistic regression roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))

""" 
Train set
Logistic regression roc-auc: 0.8123468468695123
Test set
Logistic regression roc-auc: 0.8008268347989602

We can draw the same conclusion for Logistic Regression: 
- reducing the cardinality improves the performance and generalisation of the algorithm.
 """

# Gradient Boosted Classifier
# model build on data with plenty of categories in Cabin variable

# call the model
gbc = GradientBoostingClassifier(n_estimators=300, random_state=44)

# train the model
gbc.fit(X_train[['Cabin_mapped', 'sex']], y_train)

# make predictions on train and test set
pred_train = gbc.predict_proba(X_train[['Cabin_mapped', 'sex']])
pred_test = gbc.predict_proba(X_test[['Cabin_mapped', 'sex']].fillna(0))

print('Train set')
print('Gradient Boosted Trees roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Gradient Boosted Trees roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))
""" 
Train set
Gradient Boosted Trees roc-auc: 0.862631390919749
Test set
Gradient Boosted Trees roc-auc: 0.7733117637298823 """

# model build on data with plenty of categories in Cabin variable

# call the model
gbc = GradientBoostingClassifier(n_estimators=300, random_state=44)

# train the model
gbc.fit(X_train[['Cabin_reduced', 'sex']], y_train)

# make predictions on train and test set
pred_train = gbc.predict_proba(X_train[['Cabin_reduced', 'sex']])
pred_test = gbc.predict_proba(X_test[['Cabin_reduced', 'sex']].fillna(0))

print('Train set')
print('Gradient Boosted Trees roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Gradient Boosted Trees roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))

""" 
Train set
Gradient Boosted Trees roc-auc: 0.816719415917359
Test set
Gradient Boosted Trees roc-auc: 0.8015181682429069

- Gradient Boosted trees are indeed over-fitting to the training set in those cases where the variable Cabin has 
    a lot of labels. This was expected as tree methods tend to be biased to variables with plenty of categories.

That is all for this example. I hope you enjoyed the information, and see you in the next one. """