""" 
One Hot Encoding
- One hot encoding, consists in encoding each categorical variable with different boolean variables (also called dummy variables) which take values 0 or 1, indicating if a category is present in an observation.

- For example, for the categorical variable "Gender", with labels 'female' and 'male', we can generate the boolean variable "female", which takes 1 if the person is 'female' or 0 otherwise, or we can generate the variable "male", which takes 1 if the person is 'male' and 0 otherwise.

- For the categorical variable "colour" with values 'red', 'blue' and 'green', we can create 3 new variables called "red", "blue" and "green". 
- These variables will take the value 1, if the observation is of the said colour or 0 otherwise.

Encoding into k-1 dummy variables
- Note however, that for the variable "colour", by creating 2 binary variables, say "red" and "blue", we already encode ALL the information:

- if the observation is red, it will be captured by the variable "red" (red = 1, blue = 0)
- if the observation is blue, it will be captured by the variable "blue" (red = 0, blue = 1)
- if the observation is green, it will be captured by the combination of "red" and "blue" (red = 0, blue = 0)
- We do not need to add a third variable "green" to capture that the observation is green.

- More generally, a categorical variable should be encoded by creating k-1 binary variables, where k is the number of distinct categories. 
- In the case of gender, k=2 (male / female), therefore we need to create only 1 (k - 1 = 1) binary variable. 
- In the case of colour, which has 3 different categories (k=3), we need to create 2 (k - 1 = 2) binary variables to capture all the information.

- One hot encoding into k-1 binary variables takes into account that we can use 1 less dimension and still represent the whole information: if the observation is 0 in all the binary variables, then it must be 1 in the final (not present) binary variable.

- When one hot encoding categorical variables, we create k - 1 binary variables

- Most machine learning algorithms, consider the entire data set while being fit. 
- Therefore, encoding categorical variables into k - 1 binary variables, is better, as it avoids introducing redundant information.

Exception: One hot encoding into k dummy variables
- There are a few occasions when it is better to encode variables into k dummy variables:
-- when building tree based algorithms
-- when doing feature selection by recursive algorithms
-- when interested in determine the importance of each single category
-- Tree based algorithms, as opposed to the majority of machine learning algorithms, do not evaluate the entire dataset while being trained. They randomly extract a subset of features from the data set at each node for each tree. Therefore, if we want a tree based algorithm to consider all the categories, we need to encode categorical variables into k binary variables.

- If we are planning to do feature selection by recursive elimination (or addition), or if we want to evaluate the importance of each single category of the categorical variable, then we will also need the entire set of binary variables (k) to let the machine learning model select which ones have the most predictive power.

Advantages of one hot encoding
- Straightforward to implement
- Makes no assumption about the distribution or categories of the categorical variable
- Keeps all the information of the categorical variable
- Suitable for linear models

Limitations
- Expands the feature space
- Does not add extra information while encoding
- Many dummy variables may be identical, introducing redundant information

Notes
- If our datasets contain a few highly cardinal variables, we will end up very soon with datasets with thousands of columns, which may make training of our algorithms slow, and model interpretation hard.

- In addition, many of these dummy variables may be similar to each other, since it is not unusual that 2 or more variables share the same combinations of 1 and 0s. 
- Therefore one hot encoding may introduce redundant or duplicated information even if we encode into k-1.

In this example:
We will see how to perform one hot encoding with:
- pandas
- Scikit-learn
- Feature-Engine

And the advantages and limitations of each implementation using the Titanic dataset.
"""

import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
# for one hot encoding with sklearn
from sklearn.preprocessing import OneHotEncoder
# for one hot encoding with feature-engine
# load the Titanic Dataset with a few variables for demonstration
data = pd.read_csv('dataset/titanic.csv', usecols=['sex', 'embarked', 'cabin', 'survived'])

data.head()
""" 
   survived     sex cabin embarked
0         1  female    B5        S
1         1    male   C22        S
2         0  female   C22        S
3         0    male   C22        S
4         0  female   C22        S """

# let's capture only the first letter of the cabin for this demonstration

data['cabin'] = data['cabin'].str[0]

data.head()
""" 
   survived     sex cabin embarked
0         1  female     B        S
1         1    male     C        S
2         0  female     C        S
3         0    male     C        S
4         0  female     C        S """

""" 
Encoding important
- Just like imputation, all methods of categorical encoding should be performed over the training set, and then propagated to the test set.

Why?
- Because these methods will "learn" patterns from the train data, and therefore you want to avoid leaking information and overfitting. 
- But more importantly, because we don't know whether in future / live data, we will have all the categories present in the train data, or if there will be more or less categories. 
- Therefore, we want to anticipate this uncertainty by setting the right processes right from the start. 
- We want to create transformers that learn the categories from the train set, and used those learned categories to create the dummy variables in both train and test sets. """

# let's separate into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    data[['sex', 'embarked', 'cabin']],  # predictors
    data['survived'],  # target
    test_size=0.3,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape
# ((916, 3), (393, 3))

# Let's explore the cardinality

# sex has 2 labels
X_train['sex'].unique()
# array(['female', 'male'], dtype=object)

# embarked has 3 labels and missing data
X_train['embarked'].unique()
# array(['S', 'C', 'Q', nan], dtype=object)

# cabin has 9 labels and missing data
X_train['cabin'].unique()
# array([nan, 'E', 'C', 'D', 'B', 'A', 'F', 'T', 'G'], dtype=object)

""" 
One hot encoding with pandas

Advantages
- quick
- returns pandas dataframe
- returns feature names for the dummy variables

Limitations of pandas:
- it does not preserve information from train data to propagate to test data

The pandas method get_dummies(), will create as many binary variables as categories in the variable:

If the variable colour has 3 categories in the train data, it will create 2 dummy variables. However, if the variable colour has 5 categories in the test data, it will create 4 binary variables, therefore train and test sets will end up with different number of features and will be incompatible with training and scoring using Scikit-learn.

In practice, we shouldn't be using get-dummies in our machine learning pipelines. It is however useful, for a quick data exploration. Let's look at this with examples. """

# into k dummy variables

tmp = pd.get_dummies(X_train['sex'])

tmp.head()
""" 
      female  male
501        1     0
588        1     0
402        1     0
1193       0     1
686        1     0 """

# for better visualisation let's put the dummies next to the original variable

pd.concat([X_train['sex'],pd.get_dummies(X_train['sex'])], axis=1).head()
""" 
         sex  female  male
501   female       1     0
588   female       1     0
402   female       1     0
1193    male       0     1
686   female       1     0 """

# and now let's repeat for embarked

tmp = pd.get_dummies(X_train['embarked'])

tmp.head()
""" 
      C  Q  S
501   0  0  1
588   0  0  1
402   1  0  0
1193  0  1  0
686   0  1  0 """

# and now for cabin

tmp = pd.get_dummies(X_train['cabin'])

tmp.head()
""" 
      A  B  C  D  E  F  G  T
501   0  0  0  0  0  0  0  0
588   0  0  0  0  0  0  0  0
402   0  0  0  0  0  0  0  0
1193  0  0  0  0  0  0  0  0
686   0  0  0  0  0  0  0  0
"""

# and now for all variables together: train set

tmp = pd.get_dummies(X_train)

print(tmp.shape)

tmp.head()
# (916, 13)

""" 
      sex_female  sex_male  embarked_C  ...  cabin_F  cabin_G  cabin_T
501            1         0           0  ...        0        0        0
588            1         0           0  ...        0        0        0
402            1         0           1  ...        0        0        0
1193           0         1           0  ...        0        0        0
686            1         0           0  ...        0        0        0

[5 rows x 13 columns] """

# and now for all variables together: test set

tmp = pd.get_dummies(X_test)

print(tmp.shape)
# (393, 12)

tmp.head()
""" 
      sex_female  sex_male  embarked_C  ...  cabin_E  cabin_F  cabin_G
1139           0         1           0  ...        0        0        0
533            1         0           0  ...        0        0        0
459            0         1           0  ...        0        0        0
1150           0         1           0  ...        0        0        0
393            0         1           0  ...        0        0        0

[5 rows x 12 columns] """

""" 
Notice the positives of pandas get_dummies:
- dataframe returned with feature names

And the limitations:
- The train set contains 13 dummy features, whereas the test set contains 12 features. This occurred because there was no category T in cabin in the test set.
- This will cause problems if training and scoring models with scikit-learn, because predictors require train and test sets to be of the same shape. """

""" into k -1 """

# obtaining k-1 labels: we need to indicate get_dummies to drop the first binary variable

tmp = pd.get_dummies(X_train['sex'], drop_first=True)

tmp.head()
""" 
      male
501      0
588      0
402      0
1193     1
686      0 """

# obtaining k-1 labels: we need to indicate get_dummies to drop the first binary variable

tmp = pd.get_dummies(X_train['embarked'], drop_first=True)

tmp.head()
""" 
      Q  S
501   0  1
588   0  1
402   0  0
1193  1  0
686   1  0 """

""" 
For embarked, if an observation shows 0 for Q and S, then its value must be C, the remaining category.

Caveat, this variable has missing data, so unless we encode missing data as well, all the information contained in the variable is not captured. """

# altogether: train set

tmp = pd.get_dummies(X_train, drop_first=True)

print(tmp.shape)
# (916, 10)

tmp.head()
""" 
      sex_male  embarked_Q  embarked_S  cabin_B  ...  cabin_E  cabin_F  cabin_G  cabin_T
501          0           0           1        0  ...        0        0        0        0
588          0           0           1        0  ...        0        0        0        0
402          0           0           0        0  ...        0        0        0        0
1193         1           1           0        0  ...        0        0        0        0
686          0           1           0        0  ...        0        0        0        0

[5 rows x 10 columns] """

# altogether: test set

tmp = pd.get_dummies(X_test, drop_first=True)

print(tmp.shape)
# (393, 9)

tmp.head()
""" 
      sex_male  embarked_Q  embarked_S  cabin_B  ...  cabin_D  cabin_E  cabin_F  cabin_G
1139         1           0           1        0  ...        0        0        0        0
533          0           0           1        0  ...        0        0        0        0
459          1           0           1        0  ...        0        0        0        0
1150         1           0           1        0  ...        0        0        0        0
393          1           0           1        0  ...        0        0        0        0

[5 rows x 9 columns] """

""" Bonus: get_dummies() can handle missing values """

# we can add an additional dummy variable to indicate missing data

pd.get_dummies(X_train['embarked'], drop_first=True, dummy_na=True).head()
""" 
      Q  S  NaN
501   0  1    0
588   0  1    0
402   0  0    0
1193  1  0    0
686   1  0    0 """

""" One hot encoding with Scikit-learn 

Advantages
- quick
- Creates the same number of features in train and test set

Limitations
- it returns a numpy array instead of a pandas dataframe
- it does not return the variable names, therefore inconvenient for variable exploration"""

# we create and train the encoder

encoder = OneHotEncoder(categories='auto',
                       drop='first', # to return k-1, use drop=false to return k dummies
                       sparse=False
                       ) # helps deal with rare labels

encoder.fit(X_train)

encoder.categories_
""" 
[array(['female', 'male'], dtype=object), array(['C', 'Q', 'S', nan], dtype=object), array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', nan], dtype=object)] """

# transform the train set

tmp = encoder.transform(X_train.fillna('Missing'))

pd.DataFrame(tmp).head()
"""             sex_male  embarked_Q  embarked_S  cabin_B  cabin_C  cabin_D  cabin_E  \
1139         1           0           1        0        0        0        0
533          0           0           1        0        0        0        0
459          1           0           1        0        0        0        0
1150         1           0           1        0        0        0        0
393          1           0           1        0        0        0        0

      cabin_F  cabin_G
1139        0        0
533         0        0
459         0        0
1150        0        0
393         0        0 """

# we can go ahead and transfom the test set

tmp = encoder.transform(X_test.fillna('Missing'))

pd.DataFrame(tmp).head()
""" 
      sex_male  embarked_Q  embarked_S  cabin_B  cabin_C  cabin_D  cabin_E  \        
1139         1           0           1        0        0        0        0
533          0           0           1        0        0        0        0
459          1           0           1        0        0        0        0
1150         1           0           1        0        0        0        0
393          1           0           1        0        0        0        0

      cabin_F  cabin_G
1139        0        0
533         0        0
459         0        0
1150        0        0
393         0        0 """

""" 
We can see that train and test contain the same number of features.

More details about Scikit-learn's OneHotEncoder can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html """

""" 
One hot encoding with Feature-Engine

Advantages
- quick
- returns dataframe
- returns feature names
- allows to select features to encode

Limitations
- Not sure yet. """

# https://feature-engine.readthedocs.io/en/latest/encoding/OneHotEncoder.html