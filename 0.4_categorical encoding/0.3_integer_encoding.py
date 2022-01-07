""" 
Integer Encoding
- Integer encoding consist in replacing the categories by digits from 1 to n (or 0 to n-1, depending the implementation), where n is the number of distinct categories of the variable.
- The numbers are assigned arbitrarily. This encoding method allows for quick benchmarking of machine learning models.

Advantages
- Straightforward to implement
- Does not expand the feature space

Limitations
- Does not capture any information about the categories labels
- Not suitable for linear models.

Integer encoding is better suited for non-linear methods which are able to navigate through the arbitrarily assigned digits to try and find patters that relate them to the target.

In this example:
We will see how to perform one hot encoding with:
- pandas
- Scikit-learn

And the advantages and limitations of each implementation using the House Prices dataset.

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# load dataset
data = pd.read_csv('dataset/house-prices-advanced-regression-techniques/train.csv',
    usecols=['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SalePrice'])

data.head()
"""   
    Neighborhood Exterior1st Exterior2nd  SalePrice
0      CollgCr     VinylSd     VinylSd     208500
1      Veenker     MetalSd     MetalSd     181500
2      CollgCr     VinylSd     VinylSd     223500
3      Crawfor     Wd Sdng     Wd Shng     140000
4      NoRidge     VinylSd     VinylSd     250000 """

# let's have a look at how many labels each variable has
for col in data.columns:
    print(col, ': ', len(data[col].unique()), ' labels')

""" 
Neighborhood :  25  labels
Exterior1st :  15  labels
Exterior2nd :  16  labels
SalePrice :  663  labels """

# let's explore the unique categories
data['Neighborhood'].unique()
""" array(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
       'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
       'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
       'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',
       'Blueste'], dtype=object) """

data['Exterior1st'].unique()
""" array(['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing',
       'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn',
       'Stone', 'ImStucc', 'CBlock'], dtype=object) """

data['Exterior2nd'].unique()
""" array(['VinylSd', 'MetalSd', 'Wd Shng', 'HdBoard', 'Plywood', 'Wd Sdng',
       'CmentBd', 'BrkFace', 'Stucco', 'AsbShng', 'Brk Cmn', 'ImStucc',
       'AsphShn', 'Stone', 'Other', 'CBlock'], dtype=object)
 """
 
""" 
Encoding important
- We select which digit to assign to each category using the train set, and then use those mappings in the test set.
 """
# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(
    data[['Neighborhood', 'Exterior1st', 'Exterior2nd']], # predictors
    data['SalePrice'],  # target
    test_size=0.3,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape
# ((1022, 3), (438, 3))

""" 
Integer encoding with pandas

Advantages
- quick
- returns pandas dataframe

Limitations of pandas:
- it does not preserve information from train data to propagate to test data
- We need to capture and save the mappings one by one, manually, if we are planing to use those in production.
 """

# first let's create a dictionary with the mappings of categories to numbers

ordinal_mapping = {
    k: i
    for i, k in enumerate(X_train['Neighborhood'].unique(), 0)
}

ordinal_mapping
""" {'CollgCr': 0,
 'ClearCr': 1,
 'BrkSide': 2,
 'Edwards': 3,
 'SWISU': 4,
 'Sawyer': 5,
 'Crawfor': 6,
 'NAmes': 7,
 'Mitchel': 8,
 'Timber': 9,
 'Gilbert': 10,
 'Somerst': 11,
 'MeadowV': 12,
 'OldTown': 13,
 'BrDale': 14,
 'NWAmes': 15,
 'NridgHt': 16,
 'SawyerW': 17,
 'NoRidge': 18,
 'IDOTRR': 19,
 'NPkVill': 20,
 'StoneBr': 21,
 'Blmngtn': 22,
 'Veenker': 23,
 'Blueste': 24} """

""" 
- The dictionary indicates which number will replace each category. 
- Numbers were assigned arbitrarily from 0 to n - 1 where n is the number of distinct categories.
 """

# replace the labels with the integers
X_train['Neighborhood'] = X_train['Neighborhood'].map(ordinal_mapping)
X_test['Neighborhood'] = X_test['Neighborhood'].map(ordinal_mapping)

# let's explore the result
X_train['Neighborhood'].head(10)
""" 
64      0
682     1
960     2
1384    3
1100    4
416     5
1034    6
853     7
472     3
1011    3
Name: Neighborhood, dtype: int64 """

# we can turn the previous commands into 2 functions
def find_category_mappings(df, variable):
    return {k: i for i, k in enumerate(df[variable].unique(), 0)}

def integer_encode(train, test, variable, ordinal_mapping):
    X_train[variable] = X_train[variable].map(ordinal_mapping)
    X_test[variable] = X_test[variable].map(ordinal_mapping)

# and now we run a loop over the remaining categorical variables
for variable in ['Exterior1st', 'Exterior2nd']:
    mappings = find_category_mappings(X_train, variable)
    integer_encode(X_train, X_test, variable, mappings)

# let's see the result
X_train.head()
"""       Neighborhood  Exterior1st  Exterior2nd
64               0            0            0
682              1            1            1
960              2            1            2
1384             3            2            3
1100             4            1            1 """

""" Integer Encoding with Scikit-learn """


# let's separate into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    data[['Neighborhood', 'Exterior1st', 'Exterior2nd']], # predictors
    data['SalePrice'],  # target
    test_size=0.3,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape
((1022, 3), (438, 3))

# let's create an encoder

le = LabelEncoder()
le.fit(X_train['Neighborhood'])
LabelEncoder()

# we can see the unique classes
le.classes_
""" array(['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr',
       'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
       'NAmes', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt', 'OldTown',
       'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber',
       'Veenker'], dtype=object) """

X_train['Neighborhood'] = le.transform(X_train['Neighborhood'])
X_test['Neighborhood'] = le.transform(X_test['Neighborhood'])

X_train.head()
"""       Neighborhood Exterior1st Exterior2nd
64               5     VinylSd     VinylSd
682              4     Wd Sdng     Wd Sdng
960              3     Wd Sdng     Plywood
1384             7     WdShing     Wd Shng
1100            18     Wd Sdng     Wd Sdng """

# Unfortunately, the LabelEncoder works one variable at the time. However there is a way to automate this for all the categorical variables. I took the below from this stackoverflow thread

# additional import required

from collections import defaultdict
# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(
    data[['Neighborhood', 'Exterior1st', 'Exterior2nd']], # predictors
    data['SalePrice'],  # target
    test_size=0.3,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape
# ((1022, 3), (438, 3))

d = defaultdict(LabelEncoder)

# Encoding the variable
train_transformed = X_train.apply(lambda x: d[x.name].fit_transform(x))

# # Using the dictionary to encode future data
test_transformed = X_test.apply(lambda x: d[x.name].transform(x))

test_transformed.head()
"""       Neighborhood  Exterior1st  Exterior2nd
64               5           12           13
682              4           13           14
960              3           13           10
1384             7           14           15
1100            18           13           14 """

# and to inverse transform to recover the original labels

# # Inverse the encoded
tmp = train_transformed.apply(lambda x: d[x.name].inverse_transform(x))
tmp.head()
"""      Neighborhood Exterior1st Exterior2nd
64        CollgCr     VinylSd     VinylSd
682       ClearCr     Wd Sdng     Wd Sdng
960       BrkSide     Wd Sdng     Plywood
1384      Edwards     WdShing     Wd Shng
1100        SWISU     Wd Sdng     Wd Sdng """

# Finally, there is another Scikit-learn transformer, the OrdinalEncoder, to encode multiple variables at the same time. 
# However, this transformer returns a NumPy array without column names, so it is not my favourite implementation. 
# More details here: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
