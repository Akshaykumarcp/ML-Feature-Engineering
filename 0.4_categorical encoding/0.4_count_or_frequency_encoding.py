""" 
Count or frequency encoding
- In count encoding we replace the categories by the count of the observations that show that category in the dataset. 
- Similarly, we can replace the category by the frequency or percentage of observations in the dataset. 
- That is, if 10 of our 100 observations show the colour blue, we would replace blue by 10 if doing count encoding, or by 0.1 if replacing by the frequency. 
- These techniques capture the representation of each label in a dataset, but the encoding may not necessarily be predictive of the outcome. 
- These are however, very popular encoding methods in Kaggle competitions.

The assumption of this technique is that the number observations shown by each variable is somewhat informative of the predictive power of the category.

Advantages
- Simple
- Does not expand the feature space

Disadvantages
- If 2 different categories appear the same amount of times in the dataset, that is, they appear in the same number of observations, they will be replaced by the same number: may lose valuable information.
- For example, if there are 10 observations for the category blue and 10 observations for the category red, both will be replaced by 10, and therefore, after the encoding, will appear to be the same thing.

In this example:
We will see how to perform count or frequency encoding with:
- pandas

And the advantages and limitations of each implementation using the House Prices dataset.
 """

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# load dataset
data = pd.read_csv('dataset/house-prices-advanced-regression-techniques/train.csv',
    usecols=['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SalePrice'])

data.head()
"""   Neighborhood Exterior1st Exterior2nd  SalePrice
0      CollgCr     VinylSd     VinylSd     208500
1      Veenker     MetalSd     MetalSd     181500
2      CollgCr     VinylSd     VinylSd     223500
3      Crawfor     Wd Sdng     Wd Shng     140000
4      NoRidge     VinylSd     VinylSd     250000 """

# let's have a look at how many labels each variable has
for col in data.columns:
    print(col, ': ', len(data[col].unique()), ' labels')

""" Neighborhood :  25  labels
Exterior1st :  15  labels
Exterior2nd :  16  labels
SalePrice :  663  labels """

""" 
Important
- When doing count transformation of categorical variables, it is important to calculate the count (or frequency = count / total observations) over the training set, and then use those numbers to replace the labels in the test set.
 """
# let's separate into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    data[['Neighborhood', 'Exterior1st', 'Exterior2nd']], # predictors
    data['SalePrice'],  # target
    test_size=0.3,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape
((1022, 3), (438, 3))

""" Count and Frequency encoding with pandas """

# let's obtain the counts for each one of the labels in the variable Neigbourhood

count_map = X_train['Neighborhood'].value_counts().to_dict()

count_map
""" {'NAmes': 151,
 'CollgCr': 105,
 'OldTown': 73,
 'Edwards': 71,
 'Sawyer': 61,
 'Somerst': 56,
 'Gilbert': 55,
 'NridgHt': 51,
 'NWAmes': 51,
 'SawyerW': 45,
 'BrkSide': 41,
 'Mitchel': 36,
 'Crawfor': 35,
 'Timber': 30,
 'NoRidge': 30,
 'ClearCr': 24,
 'IDOTRR': 24,
 'SWISU': 18,
 'StoneBr': 16,
 'Blmngtn': 12,
 'MeadowV': 12,
 'BrDale': 10,
 'NPkVill': 7,
 'Veenker': 6,
 'Blueste': 2} """

# The dictionary contains the number of observations per category in Neighbourhood.

# replace the labels with the counts
X_train['Neighborhood'] = X_train['Neighborhood'].map(count_map)
X_test['Neighborhood'] = X_test['Neighborhood'].map(count_map)

# let's explore the result
X_train['Neighborhood'].head(10)
""" 64      105
682      24
960      41
1384     71
1100     18
416      61
1034     35
853     151
472      71
1011     71
Name: Neighborhood, dtype: int64 """

# if instead of the count we would like the frequency
# we need only divide the count by the total number of observations:

frequency_map = (X_train['Exterior1st'].value_counts() / len(X_train) ).to_dict()
frequency_map
""" {'VinylSd': 0.3561643835616438,
 'HdBoard': 0.149706457925636,
 'Wd Sdng': 0.14481409001956946,
 'MetalSd': 0.1350293542074364,
 'Plywood': 0.08414872798434442,
 'CemntBd': 0.03816046966731898,
 'BrkFace': 0.03424657534246575,
 'WdShing': 0.02054794520547945,
 'Stucco': 0.016634050880626222,
 'AsbShng': 0.014677103718199608,
 'Stone': 0.0019569471624266144,
 'BrkComm': 0.0009784735812133072,
 'ImStucc': 0.0009784735812133072,
 'CBlock': 0.0009784735812133072,
 'AsphShn': 0.0009784735812133072} """

# replace the labels with the frequencies

X_train['Exterior1st'] = X_train['Exterior1st'].map(frequency_map)
X_test['Exterior1st'] = X_test['Exterior1st'].map(frequency_map)

# We can then put these commands into 2 functions as we did in the previous 3 examples, and loop over all the categorical variables.
# If you don't know how to do this, please check any of the previous examples.