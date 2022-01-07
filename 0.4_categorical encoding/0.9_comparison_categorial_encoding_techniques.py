""" 
Comparison of Categorical Variable Encodings
- In this example, we will compare the performance of the different feature categorical encoding 
    techniques we learned so far.

We will compare:
- One hot encoding
- Replacing labels by the count
- Target Ordering mean and integer encoding labels according to the target
- Target Mean Encoding
- Target WoE

Using the titanic dataset """

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# let's load the titanic dataset

# we will only use these columns in the demo
cols = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex', 'cabin', 'embarked', 'survived']

data = pd.read_csv('dataset/titanic.csv', usecols=cols)

data.head()
""" 
   pclass  survived     sex      age  sibsp  parch      fare cabin embarked
0       1         1  female  29.0000      0      0  211.3375    B5        S
1       1         1    male   0.9167      1      2  151.5500   C22        S
2       1         0  female   2.0000      1      2  151.5500   C22        S
3       1         0    male  30.0000      1      2  151.5500   C22        S
4       1         0  female  25.0000      1      2  151.5500   C22        S """

# let's check for missing data
""" 
data.isnull().sum()
pclass         0
survived       0
sex            0
age          263
sibsp          0
parch          0
fare           1
cabin       1014
embarked       2
dtype: int64 """

# Drop observations with NA in Fare and embarked
data.dropna(subset=['fare', 'embarked'], inplace=True)

# Now we extract the first letter of the cabin
data['cabin'] = data['cabin'].astype(str).str[0]

data.head()
"""    
pclass  survived     sex      age  sibsp  parch      fare cabin embarked
0       1         1  female  29.0000      0      0  211.3375     B        S
1       1         1    male   0.9167      1      2  151.5500     C        S
2       1         0  female   2.0000      1      2  151.5500     C        S
3       1         0    male  30.0000      1      2  151.5500     C        S
4       1         0  female  25.0000      1      2  151.5500     C        S """

# drop observations with cabin = T, they are too few
data = data[data['cabin'] != 'T']

# Let's divide into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels='survived', axis=1),  # predictors
    data['survived'],  # target
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape
# ((913, 8), (392, 8))

# Let's replace null values in numerical variables by the mean
def impute_na(df, variable, value):
    df[variable].fillna(value, inplace=True)

impute_na(X_test, 'age', X_train['age'].mean())
impute_na(X_train, 'age',  X_train['age'].mean())

# note how I impute first the test set, this way the value of
# the median used will be the same for both train and test

X_train.head()
"""       
        pclass     sex       age  sibsp  parch     fare cabin embarked
402        2  female  30.00000      1      0  13.8583     n        C
698        3    male  18.00000      0      0   8.6625     n        S
1291       3    male  29.79847      0      0   8.7125     n        S
1229       3    male  27.00000      0      0   8.6625     n        S
118        1    male  29.79847      0      0  26.5500     D        S """

# let's check that we have no missing data after NA imputation
X_train.isnull().sum(), X_test.isnull().sum()
""" 
(pclass      0
 sex         0
 age         0
 sibsp       0
 parch       0
 fare        0
 cabin       0
 embarked    0
 dtype: int64, pclass      0

 sex         0
 age         0
 sibsp       0
 parch       0
 fare        0
 cabin       0
 embarked    0
 dtype: int64) """

""" One Hot Encoding """

def get_OHE(df):
    df_OHE = pd.concat(
        [df[['pclass', 'age', 'sibsp', 'parch', 'fare']],
         pd.get_dummies(df[['sex', 'cabin', 'embarked']], drop_first=True)],
        axis=1)
    return df_OHE

X_train_OHE = get_OHE(X_train)
X_test_OHE = get_OHE(X_test)

X_train_OHE.head()
""" 
      pclass       age  sibsp  parch     fare  sex_male  cabin_B  cabin_C  cabin_D  cabin_E  cabin_F  cabin_G  cabin_n  embarked_Q  embarked_S
402        2  30.00000      1      0  13.8583         0        0        0        0        0        0        0        1           0           0
698        3  18.00000      0      0   8.6625         1        0        0        0        0        0        0        1           0           1
1291       3  29.79847      0      0   8.7125         1        0        0        0        0        0        0        1           0           1
1229       3  27.00000      0      0   8.6625         1        0        0        0        0        0        0        1           0           1
118        1  29.79847      0      0  26.5500         1        0        0        1        0        0        0        0           0           1 """

X_test_OHE.head()
""" 
      pclass       age  sibsp  parch     fare  sex_male  cabin_B  cabin_C  cabin_D  cabin_E  cabin_F  cabin_G  cabin_n  embarked_Q  embarked_S
586        2  29.00000      1      0  26.0000         0        0        0        0        0        0        0        1           0           1
200        1  46.00000      0      0  75.2417         1        0        1        0        0        0        0        0           0           0
831        3  40.00000      1      6  46.9000         1        0        0        0        0        0        0        1           0           1
1149       3  29.79847      0      0   7.7208         0        0        0        0        0        0        0        1           1           0
393        2  25.00000      0      0  31.5000         1        0        0        0        0        0        0        1           0           1 """

""" Count encoding """

def categorical_to_counts(df_train, df_test):
    # make a temporary copy of the original dataframes
    df_train_temp = df_train.copy()
    df_test_temp = df_test.copy()

    for col in ['sex', 'cabin', 'embarked']:

        # make dictionary mapping category to counts
        counts_map = df_train_temp[col].value_counts().to_dict()

        # remap the labels to their counts
        df_train_temp[col] = df_train_temp[col].map(counts_map)
        df_test_temp[col] = df_test_temp[col].map(counts_map)

    return df_train_temp, df_test_temp

X_train_count, X_test_count = categorical_to_counts(X_train, X_test)

X_train_count.head()
""" 
      pclass  sex       age  sibsp  parch     fare  cabin  embarked
402        2  326  30.00000      1      0  13.8583    702       184
698        3  587  18.00000      0      0   8.6625    702       647
1291       3  587  29.79847      0      0   8.7125    702       647
1229       3  587  27.00000      0      0   8.6625    702       647
118        1  587  29.79847      0      0  26.5500     33       647 """

""" Target Ordering mean and integer encoding labels according to the target """

def categories_to_ordered(df_train, df_test, y_train, y_test):

    # make a temporary copy of the datasets
    df_train_temp = pd.concat([df_train, y_train], axis=1).copy()
    df_test_temp = pd.concat([df_test, y_test], axis=1).copy()

    for col in ['sex', 'cabin', 'embarked']:

        # order categories according to target mean
        ordered_labels = df_train_temp.groupby(
            [col])['survived'].mean().sort_values().index

        # create the dictionary to map the ordered labels to an ordinal number
        ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}

        # remap the categories  to these ordinal numbers
        df_train_temp[col] = df_train[col].map(ordinal_label)
        df_test_temp[col] = df_test[col].map(ordinal_label)

    # remove the target
    df_train_temp.drop(['survived'], axis=1, inplace=True)
    df_test_temp.drop(['survived'], axis=1, inplace=True)

    return df_train_temp, df_test_temp

X_train_ordered, X_test_ordered = categories_to_ordered(X_train, X_test, y_train, y_test)

X_train_ordered.head()
""" 
      pclass  sex       age  sibsp  parch     fare  cabin  embarked
402        2    1  30.00000      1      0  13.8583      0         2
698        3    0  18.00000      0      0   8.6625      0         0
1291       3    0  29.79847      0      0   8.7125      0         0
1229       3    0  27.00000      0      0   8.6625      0         0
118        1    0  29.79847      0      0  26.5500      5         0 """

""" Target Mean Encoding """

def categories_to_mean(df_train, df_test, y_train, y_test):

    # make a temporary copy of the datasets
    df_train_temp = pd.concat([df_train, y_train], axis=1).copy()
    df_test_temp = pd.concat([df_test, y_test], axis=1).copy()

    for col in ['sex', 'cabin', 'embarked']:

        # calculate mean target per category
        ordered_labels = df_train_temp.groupby([col])['survived'].mean().to_dict()

        # remap the categories to target mean
        df_train_temp[col] = df_train[col].map(ordered_labels)
        df_test_temp[col] = df_test[col].map(ordered_labels)

    # remove the target
    df_train_temp.drop(['survived'], axis=1, inplace=True)
    df_test_temp.drop(['survived'], axis=1, inplace=True)

    return df_train_temp, df_test_temp

X_train_mean, X_test_mean = categories_to_mean(X_train, X_test, y_train, y_test)

X_train_mean.head()
""" 
      pclass       sex       age  sibsp  parch     fare     cabin  embarked
402        2  0.730061  30.00000      1      0  13.8583  0.292023  0.516304
698        3  0.173765  18.00000      0      0   8.6625  0.292023  0.332303
1291       3  0.173765  29.79847      0      0   8.7125  0.292023  0.332303
1229       3  0.173765  27.00000      0      0   8.6625  0.292023  0.332303
118        1  0.173765  29.79847      0      0  26.5500  0.696970  0.332303 """

""" Target WoE """

def categories_to_woe(df_train, df_test, y_train, y_test):

    # make a temporary copy of the datasets
    df_train_temp = pd.concat([df_train, y_train], axis=1).copy()
    df_test_temp = pd.concat([df_test, y_test], axis=1).copy()

    for col in ['sex', 'cabin', 'embarked']:

        # create df containing the different parts of the WoE equation
        # prob survived =1
        prob_df = pd.DataFrame(df_train_temp.groupby([col])['survived'].mean())

        # prob survived = 0
        prob_df['died'] = 1-prob_df.survived

        # calculate WoE
        prob_df['WoE'] = np.log(prob_df.survived/prob_df.died)

        # capture woe in dictionary
        woe = prob_df['WoE'].to_dict()

        # re-map the labels to WoE
        df_train_temp[col] = df_train[col].map(woe)
        df_test_temp[col] = df_test[col].map(woe)

    # drop the target
    df_train_temp.drop(['survived'], axis=1, inplace=True)
    df_test_temp.drop(['survived'], axis=1, inplace=True)

    return df_train_temp, df_test_temp

X_train_woe, X_test_woe = categories_to_woe(X_train, X_test, y_train, y_test)

X_train_woe.head()
""" 
      pclass       sex       age  sibsp  parch     fare     cabin  embarked
402        2  0.994934  30.00000      1      0  13.8583 -0.885580  0.065241
698        3 -1.559176  18.00000      0      0   8.6625 -0.885580 -0.697788
1291       3 -1.559176  29.79847      0      0   8.7125 -0.885580 -0.697788
1229       3 -1.559176  27.00000      0      0   8.6625 -0.885580 -0.697788
118        1 -1.559176  29.79847      0      0  26.5500  0.832909 -0.697788 """

""" Random Forest Performance """

# create a function to build random forests and compare performance in train and test set

def run_randomForests(X_train, X_test, y_train, y_test):

    rf = RandomForestClassifier(n_estimators=50, random_state=39, max_depth=3)
    rf.fit(X_train, y_train)

    print('Train set')
    pred = rf.predict_proba(X_train)
    print(
        'Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))

    print('Test set')
    pred = rf.predict_proba(X_test)
    print(
        'Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))

# OHE
run_randomForests(X_train_OHE, X_test_OHE, y_train, y_test)
""" 
Train set
Random Forests roc-auc: 0.8488938507340109
Test set
Random Forests roc-auc: 0.8072730715135779 """

# counts

run_randomForests(X_train_count, X_test_count, y_train, y_test)
""" 
Train set
Random Forests roc-auc: 0.8654552920644698
Test set
Random Forests roc-auc: 0.8194309206967434 """

# ordered labels

run_randomForests(X_train_ordered, X_test_ordered, y_train, y_test)
""" 
Train set
Random Forests roc-auc: 0.8669027820552304
Test set
Random Forests roc-auc: 0.8219733852645245 """

# mean encoding

run_randomForests(X_train_mean, X_test_mean, y_train, y_test)

""" 
Train set
Random Forests roc-auc: 0.867010573863053
Test set
Random Forests roc-auc: 0.8207562479714378 """

# woe
run_randomForests(X_train_woe, X_test_woe, y_train, y_test)
"""
Train set
Random Forests roc-auc: 0.867010573863053
Test set
Random Forests roc-auc: 0.8207562479714378 """

""" 
OBSERVATION:
- Comparing the roc_auc values on the test sets, we can see that one hot encoding has the worse performance. 
- This makes sense because trees do not perform well in datasets with big feature spaces.

- The remaining encodings returned similar performances. 
- This also makes sense, because trees are non-linear models, so target guided encodings may not necessarily improve the model performance
"""
""" Logistic Regression Performance """

def run_logistic(X_train, X_test, y_train, y_test):

    # function to train and test the performance of logistic regression
    logit = LogisticRegression(random_state=44, C=0.01)
    logit.fit(X_train, y_train)

    print('Train set')
    pred = logit.predict_proba(X_train)
    print(
        'Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))

    print('Test set')
    pred = logit.predict_proba(X_test)
    print(
        'Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))

# OHE
run_logistic(X_train_OHE, X_test_OHE, y_train, y_test)
""" 
Train set
Logistic Regression roc-auc: 0.8287932450467097
Test set
Logistic Regression roc-auc: 0.8013902412636589
 """

# counts

run_logistic(X_train_count, X_test_count, y_train, y_test)
""" 
Train set
Logistic Regression roc-auc: 0.7910866440817166
Test set
Logistic Regression roc-auc: 0.7401006166828952 """

# ordered labels
run_logistic(X_train_ordered, X_test_ordered, y_train, y_test)
""" 
Train set
Logistic Regression roc-auc: 0.8223975977825686
Test set
Logistic Regression roc-auc: 0.8006870063832089 """

# mean encoding
run_logistic(X_train_mean, X_test_mean, y_train, y_test)
""" 
Train set
Logistic Regression roc-auc: 0.7791217534134072
Test set
Logistic Regression roc-auc: 0.7481878178080709 """

# woe
run_logistic(X_train_woe, X_test_woe, y_train, y_test)
""" 
Train set
Logistic Regression roc-auc: 0.8508546350477364
Test set
Logistic Regression roc-auc: 0.8204857730174184 """

""" 
OBSERVATION:
- For Logistic regression, the best performances are obtained with one hot encoding, as it preserves linear relationships with variables and target, and also with weight of evidence, and ordered encoding.

- Note however how count encoding, returns the worse performance as it does not create a monotonic relationship between variables and target, and 
- in this case, mean target encoding is probably causing over-fitting. """