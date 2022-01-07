 """ 
One Hot Encoding {OHE} of Frequent Categories
- We learned in previous examples that high cardinality and rare labels may result in certain categories appearing only in the train set, therefore causing over-fitting, or only in the test set, and then our models wouldn't know how to score those observations.
- We also learned in the previous example on one hot encoding, that if categorical variables contain multiple labels, then by re-encoding them with dummy variables we will expand the feature space dramatically.

In order to avoid these complications, we can create dummy variables only for the most frequent categories

This procedure is also called one hot encoding of top categories.

- In fact, in the winning solution of the KDD 2009 cup: "Winning the KDD Cup Orange Challenge with Ensemble Selection", the authors limit one hot encoding to the 10 most frequent labels of the variable. This means that they would make one binary variable for each of the 10 most frequent labels only.
- OHE of frequent or top categories is equivalent to grouping all the remaining categories under a new category. We will have a better look at grouping rare values into a new category in a later examples in this section.

Advantages of OHE of top categories
- Straightforward to implement
- Does not require hrs of variable exploration
- Does not expand massively the feature space
- Suitable for linear models

Limitations
- Does not add any information that may make the variable more predictive
- Does not keep the information of the ignored labels

Often, categorical variables show a few dominating categories while the remaining labels add little information. Therefore, OHE of top categories is a simple and useful technique.

Note
- The number of top variables is set arbitrarily. In the KDD competition the authors selected 10, but it could have been 15 or 5 as well. 
- This number can be chosen arbitrarily or derived from data exploration.

In this example:
We will see how to perform one hot encoding with:
- pandas and NumPy

And the advantages and limitations of these implementations using the House Prices dataset. """

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# load dataset

data = pd.read_csv(
    'dataset/house-prices-advanced-regression-techniques/train.csv',
    usecols=['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SalePrice'])

data.head()
""" 
  Neighborhood Exterior1st Exterior2nd  SalePrice
0      CollgCr     VinylSd     VinylSd     208500
1      Veenker     MetalSd     MetalSd     181500
2      CollgCr     VinylSd     VinylSd     223500
3      Crawfor     Wd Sdng     Wd Shng     140000
4      NoRidge     VinylSd     VinylSd     250000 """

for col in data.columns:
    print(col, ': ', len(data[col].unique()), ' labels')

""" 
Neighborhood :  25  labels
Exterior1st :  15  labels
Exterior2nd :  16  labels
SalePrice :  663  labels """

# let's explore the unique categories
data['Neighborhood'].unique()
""" 
array(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
       'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
       'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
       'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',
       'Blueste'], dtype=object) """

data['Exterior1st'].unique()
""" 
array(['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing',
       'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn',
       'Stone', 'ImStucc', 'CBlock'], dtype=object) """

data['Exterior2nd'].unique()
""" 
array(['VinylSd', 'MetalSd', 'Wd Shng', 'HdBoard', 'Plywood', 'Wd Sdng',
       'CmentBd', 'BrkFace', 'Stucco', 'AsbShng', 'Brk Cmn', 'ImStucc',
       'AsphShn', 'Stone', 'Other', 'CBlock'], dtype=object) """

""" 
Encoding important
- It is important to select the top or most frequent categories based of the train data. 
- Then, we will use those top categories to encode the variables in the test data as well """

# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(
    data[['Neighborhood', 'Exterior1st', 'Exterior2nd']],  # predictors
    data['SalePrice'],  # target
    test_size=0.3,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape
# ((1022, 3), (438, 3))

# let's first examine how OHE expands the feature space

pd.get_dummies(X_train, drop_first=True).shape
# (1022, 53)

""" 
observation:
- From the initial 3 categorical variables, we end up with 53 variables.
- These numbers are still not huge, and in practice we could work with them relatively easily. However, in real-life datasets, categorical variables can be highly cardinal, and with OHE we can end up with datasets with thousands of columns. """

""" 
OHE with pandas and NumPy

Advantages
- quick
- returns pandas dataframe
- returns feature names for the dummy variables

Limitations:
- it does not preserve information from train data to propagate to test data """

# let's find the top 10 most frequent categories for the variable 'Neighborhood'

X_train['Neighborhood'].value_counts().sort_values(ascending=False).head(10)
""" NAmes      151
CollgCr    105
OldTown     73
Edwards     71
Sawyer      61
Somerst     56
Gilbert     55
NridgHt     51
NWAmes      51
SawyerW     45
Name: Neighborhood, dtype: int64 """

# let's make a list with the most frequent categories of the variable

top_10 = [
    x for x in X_train['Neighborhood'].value_counts().sort_values(
        ascending=False).head(10).index
]

top_10
# ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Sawyer', 'Somerst', 'Gilbert', 'NridgHt', 'NWAmes', 'SawyerW']

# and now we make the 10 binary variables
import numpy as np
for label in top_10:
    X_train['Neighborhood' + '_' + label] = np.where(X_train['Neighborhood'] == label, 1, 0)
    
    X_test['Neighborhood' + '_' + label] = np.where(X_test['Neighborhood'] == label, 1, 0)

# let's visualise the result
X_train[['Neighborhood'] + ['Neighborhood'+'_'+c for c in top_10]].head(10)
"""      Neighborhood  Neighborhood_NAmes  Neighborhood_CollgCr  \
64        CollgCr                   0                     1
682       ClearCr                   0                     0
960       BrkSide                   0                     0
1384      Edwards                   0                     0
1100        SWISU                   0                     0
416        Sawyer                   0                     0
1034      Crawfor                   0                     0
853         NAmes                   1                     0
472       Edwards                   0                     0
1011      Edwards                   0                     0

      Neighborhood_OldTown  Neighborhood_Edwards  Neighborhood_Sawyer  \
64                       0                     0                    0
682                      0                     0                    0
960                      0                     0                    0
1384                     0                     1                    0
1100                     0                     0                    0
416                      0                     0                    1
1034                     0                     0                    0
853                      0                     0                    0
472                      0                     1                    0
1011                     0                     1                    0

      Neighborhood_Somerst  Neighborhood_Gilbert  Neighborhood_NWAmes  \
64                       0                     0                    0
682                      0                     0                    0
960                      0                     0                    0
1384                     0                     0                    0
1100                     0                     0                    0
416                      0                     0                    0
1034                     0                     0                    0
853                      0                     0                    0
472                      0                     0                    0
1011                     0                     0                    0

      Neighborhood_NridgHt  Neighborhood_SawyerW
64                       0                     0
682                      0                     0
960                      0                     0
1384                     0                     0
1100                     0                     0
416                      0                     0
1034                     0                     0
853                      0                     0
472                      0                     0
1011                     0                     0 """


# we can turn the previous commands into 2 functions

def calculate_top_categories(df, variable, how_many=10):
    return [
        x for x in df[variable].value_counts().sort_values(
            ascending=False).head(how_many).index
    ]


def one_hot_encode(train, test, variable, top_x_labels):

    for label in top_x_labels:
        train[variable + '_' + label] = np.where(train['Neighborhood'] == label, 1, 0)
        test[variable + '_' + label] = np.where(test['Neighborhood'] == label,1, 0)

# and now we run a loop over the remaining categorical variables

for variable in ['Exterior1st', 'Exterior2nd']:
    top_categories = calculate_top_categories(X_train, variable, how_many=10)
    one_hot_encode(X_train, X_test, variable, top_categories)

# let's see the result

X_train.head()
"""      Neighborhood Exterior1st Exterior2nd  Neighborhood_NAmes  \
64        CollgCr     VinylSd     VinylSd                   0
682       ClearCr     Wd Sdng     Wd Sdng                   0
960       BrkSide     Wd Sdng     Plywood                   0
1384      Edwards     WdShing     Wd Shng                   0
1100        SWISU     Wd Sdng     Wd Sdng                   0

      Neighborhood_CollgCr  Neighborhood_OldTown  Neighborhood_Edwards  \
64                       1                     0                     0
682                      0                     0                     0
960                      0                     0                     0
1384                     0                     0                     1
1100                     0                     0                     0

      Neighborhood_Sawyer  Neighborhood_Somerst  Neighborhood_Gilbert  \
64                      0                     0                     0
682                     0                     0                     0
960                     0                     0                     0
1384                    0                     0                     0
1100                    0                     0                     0

      Neighborhood_NWAmes  Neighborhood_NridgHt  Neighborhood_SawyerW  \
64                      0                     0                     0
682                     0                     0                     0
960                     0                     0                     0
1384                    0                     0                     0
1100                    0                     0                     0

      Exterior1st_VinylSd  Exterior1st_HdBoard  Exterior1st_Wd Sdng  \
64                      0                    0                    0
682                     0                    0                    0
960                     0                    0                    0
1384                    0                    0                    0
1100                    0                    0                    0

      Exterior1st_MetalSd  Exterior1st_Plywood  Exterior1st_CemntBd  \
64                      0                    0                    0
682                     0                    0                    0
960                     0                    0                    0
1384                    0                    0                    0
1100                    0                    0                    0

      Exterior1st_BrkFace  Exterior1st_WdShing  Exterior1st_Stucco  \
64                      0                    0                   0
682                     0                    0                   0
960                     0                    0                   0
1384                    0                    0                   0
1100                    0                    0                   0

      Exterior1st_AsbShng  Exterior2nd_VinylSd  Exterior2nd_Wd Sdng  \
64                      0                    0                    0
682                     0                    0                    0
960                     0                    0                    0
1384                    0                    0                    0
1100                    0                    0                    0

      Exterior2nd_HdBoard  Exterior2nd_MetalSd  Exterior2nd_Plywood  \
64                      0                    0                    0
682                     0                    0                    0
960                     0                    0                    0
1384                    0                    0                    0
1100                    0                    0                    0

      Exterior2nd_CmentBd  Exterior2nd_Wd Shng  Exterior2nd_BrkFace  \
64                      0                    0                    0
682                     0                    0                    0
960                     0                    0                    0
1384                    0                    0                    0
1100                    0                    0                    0

      Exterior2nd_AsbShng  Exterior2nd_Stucco
64                      0                   0
682                     0                   0
960                     0                   0
1384                    0                   0
1100                    0                   0 """

""" 
Observation:
- Note how we now have 30 additional dummy variables instead of the 53 that we would have had if we had created dummies for all categories.
 """
