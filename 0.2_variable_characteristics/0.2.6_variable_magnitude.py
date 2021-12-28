""" 
Variable magnitude

Does the magnitude of the variable matter?
    In Linear Regression models, the scale of variables used to estimate the output matters. 
    Linear models are of the type y = w x + b, where the regression coefficient w represents the expected change 
    in y for a one unit change in x (the predictor). 
    Thus, the magnitude of w is partly determined by the magnitude of the units being used for x. 
    If x is a distance variable, just changing the scale from kilometers to miles will cause a change in the 
    magnitude of the coefficient.

- In addition, in situations where we estimate the outcome y by contemplating multiple predictors x1, x2, ...xn, 
    predictors with greater numeric ranges dominate over those with smaller numeric ranges.

- Gradient descent converges faster when all the predictors (x1 to xn) are within a similar scale, therefore having 
    features in a similar scale is useful for Neural Networks as well as.

- In Support Vector Machines, feature scaling can decrease the time to find the support vectors.

- Finally, methods using Euclidean distances or distances in general are also affected by the magnitude of the features, 
    as Euclidean distance is sensitive to variations in the magnitude or scales of the predictors. 
    Therefore feature scaling is required for methods that utilise distance calculations like k-nearest neighbours (KNN)
    and k-means clustering.

In summary:
Magnitude matters because:
- The regression coefficient is directly influenced by the scale of the variable
- Variables with bigger magnitude / value range dominate over the ones with smaller magnitude / value range
- Gradient descent converges faster when features are on similar scales
- Feature scaling helps decrease the time to find support vectors for SVMs
- Euclidean distances are sensitive to feature magnitude.

The machine learning models affected by the magnitude of the feature are:
- Linear and Logistic Regression
- Neural Networks
- Support Vector Machines
- KNN
- K-means clustering
- Linear Discriminant Analysis (LDA)
- Principal Component Analysis (PCA)

Machine learning models insensitive to feature magnitude are the ones based on Trees:
- Classification and Regression Trees
- Random Forests
-Gradient Boosted Trees

===================================================================================================

In this example:
- We will study the effect of feature magnitude on the performance of different machine learning algorithms.
- We will use the Titanic dataset. """

import pandas as pd
import numpy as np
# import several machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
# to scale the features
from sklearn.preprocessing import MinMaxScaler
# to evaluate performance and separate into # train and test set
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# Load data with numerical variables only
# load numerical variables of the Titanic Dataset

data = pd.read_csv('dataset/titanic.csv',usecols=['pclass', 'age', 'fare', 'survived'])
data.head()
""" 
   pclass  survived      age      fare
0       1         1  29.0000  211.3375
1       1         1   0.9167  151.5500
2       1         0   2.0000  151.5500
3       1         0  30.0000  151.5500
4       1         0  25.0000  151.5500 """


# let's have a look at the values of those variables
# to get an idea of the feature magnitudes

data.describe()
"""         pclass     survived          age         fare
count  1309.000000  1309.000000  1046.000000  1308.000000
mean      2.294882     0.381971    29.881135    33.295479
std       0.837836     0.486055    14.413500    51.758668
min       1.000000     0.000000     0.166700     0.000000
25%       2.000000     0.000000    21.000000     7.895800
50%       3.000000     0.000000    28.000000    14.454200
75%       3.000000     1.000000    39.000000    31.275000
max       3.000000     1.000000    80.000000   512.329200 

Observation: 
- We can see that Fare varies between 0 and 512, Age between 0 and 80, and Class between 0 and 3. 
    So the variables have different magnitude.
"""

# let's now calculate the range
for col in ['pclass', 'age', 'fare']:
    print(col, 'range: ', data[col].max() - data[col].min())
""" 
pclass range:  2
age range:  79.8333
fare range:  512.3292

Observation: 
- The range of values that each variable can take are quite different. """

# let's separate into training and testing set the titanic dataset contains missing information
# so for this example, I will fill those in with 0s

X_train, X_test, y_train, y_test = train_test_split(
    data[['pclass', 'age', 'fare']].fillna(0),
    data.survived,
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape
# ((916, 3), (393, 3))

""" 
Feature Scaling
- For this example, I will scale the features between 0 and 1, using the MinMaxScaler from scikit-learn. 
    To learn more about this scaling visit the Scikit-Learn website

The transformation is given by:
- X_rescaled = X - X.min() / (X.max - X.min()
- And to transform the re-scaled features back to their original magnitude:
- X = X_rescaled * (max - min) + min

There is a dedicated section to feature scaling in upcoming example, 
where I will explain this and other scaling techniques in more detail. 

For now, let's carry on with the demonstration.
 """
# scale the features between 0 and 1.

# cal the scaler
scaler = MinMaxScaler()

# fit the scaler
scaler.fit(X_train)

# re scale the datasets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#let's have a look at the scaled training dataset
print('Mean: ', X_train_scaled.mean(axis=0))
print('Standard Deviation: ', X_train_scaled.std(axis=0))
print('Minimum value: ', X_train_scaled.min(axis=0))
print('Maximum value: ', X_train_scaled.max(axis=0))

"""
Mean:  [0.64628821 0.33048359 0.06349833]
Standard Deviation:  [0.42105785 0.23332045 0.09250036]
Minimum value:  [0. 0. 0.]
Maximum value:  [1. 1. 1.]

Now, the maximum values for all the features is 1, and the minimum value is zero, as expected. 
So they are in a more similar scale.
 """

""" Logistic Regression
Let's evaluate the effect of feature scaling in a Logistic Regression.
 """
# model build on unscaled variables

# call the model
logit = LogisticRegression(
    random_state=44,
    C=1000,  # c big to avoid regularization
    solver='lbfgs')

# train the model
logit.fit(X_train, y_train)

# evaluate performance
print('Train set')
pred = logit.predict_proba(X_train)
print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))
print('Test set')
pred = logit.predict_proba(X_test)
print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))
""" 
Train set
Logistic Regression roc-auc: 0.6793181006244372
Test set
Logistic Regression roc-auc: 0.7175488081411426 """

# let's look at the coefficients
logit.coef_
# array([[-0.71428242, -0.00923013,  0.00425235]])
# model built on scaled variables

# call the model
logit = LogisticRegression(
    random_state=44,
    C=1000,  # c big to avoid regularization
    solver='lbfgs')

# train the model using the re-scaled data
logit.fit(X_train_scaled, y_train)

# evaluate performance
print('Train set')
pred = logit.predict_proba(X_train_scaled)
print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))
print('Test set')
pred = logit.predict_proba(X_test_scaled)
print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))
""" 
Train set
Logistic Regression roc-auc: 0.6793281640744896
Test set
Logistic Regression roc-auc: 0.7175488081411426 """

logit.coef_
# array([[-1.42875872, -0.68293349,  2.17646757]])

""" 
- We observe that the performance of logistic regression did not change when using the datasets with the 
    features scaled (compare roc-auc values for train and test set for models with and without feature scaling).

- However, when looking at the coefficients we do see a big difference in the values. 
    This is because the magnitude of the variable was affecting the coefficients. 
    After scaling, all 3 variables have the relatively the same effect (coefficient) towards survival, 
    whereas before scaling, we would be inclined to think that PClass was driving the Survival outcome.

Support Vector Machines
 """# model build on unscaled variables

# call the model
SVM_model = SVC(random_state=44, probability=True, gamma='auto')

#  train the model
SVM_model.fit(X_train, y_train)

# evaluate performance
print('Train set')
pred = SVM_model.predict_proba(X_train)
print('SVM roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))
print('Test set')
pred = SVM_model.predict_proba(X_test)
print('SVM roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))
""" 
Train set
SVM roc-auc: 0.8824010385480454
Test set
SVM roc-auc: 0.6617443725457663 """

# model built on scaled variables

# call the model
SVM_model = SVC(random_state=44, probability=True, gamma='auto')

# train the model
SVM_model.fit(X_train_scaled, y_train)

# evaluate performance
print('Train set')
pred = SVM_model.predict_proba(X_train_scaled)
print('SVM roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))
print('Test set')
pred = SVM_model.predict_proba(X_test_scaled)
print('SVM roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))
"""
Train set
SVM roc-auc: 0.6784778025450466
Test set
SVM roc-auc: 0.683894696089818

- Feature scaling improved the performance of the support vector machine. After feature scaling the model is no 
    longer over-fitting to the training set (compare the roc-auc of 0.881 for the model on unscaled features vs 
    the roc-auc of 0.68). 
    In addition, the roc-auc for the testing set increased as well (0.66 vs 0.68).

K-Nearest Neighbours """
#model built on unscaled features

# call the model
KNN = KNeighborsClassifier(n_neighbors=5)

# train the model
KNN.fit(X_train, y_train)

# evaluate performance
print('Train set')
pred = KNN.predict_proba(X_train)
print('KNN roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = KNN.predict_proba(X_test)
print('KNN roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
""" 
Train set
KNN roc-auc: 0.8131141849360215
Test set
KNN roc-auc: 0.6947901111664178 """

# model built on scaled

# call the model
KNN = KNeighborsClassifier(n_neighbors=5)

# train the model
KNN.fit(X_train_scaled, y_train)

# evaluate performance
print('Train set')
pred = KNN.predict_proba(X_train_scaled)
print('KNN roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = KNN.predict_proba(X_test_scaled)
print('KNN roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
""" 
Train set
KNN roc-auc: 0.826928785995703
Test set
KNN roc-auc: 0.7232453957192633

- We observe for KNN as well that feature scaling improved the performance of the model. 
- The model built on unscaled features shows a better generalisation, with a higher roc-auc for the testing set 
    (0.72 vs 0.69 for model built on unscaled features).

- Both KNN methods are over-fitting to the train set. 
- Thus, we would need to change the parameters of the model or use less features to try and decrease over-fitting, 
    which exceeds the purpose of this example.

Random Forests """
# model built on unscaled features

# call the model
rf = RandomForestClassifier(n_estimators=200, random_state=39)

# train the model
rf.fit(X_train, y_train)

# evaluate performance
print('Train set')
pred = rf.predict_proba(X_train)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))
print('Test set')
pred = rf.predict_proba(X_test)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))
""" Train set
Random Forests roc-auc: 0.9866810238554083
Test set
Random Forests roc-auc: 0.7326751838946961 """
# model built in scaled features

# call the model
rf = RandomForestClassifier(n_estimators=200, random_state=39)

# train the model
rf.fit(X_train_scaled, y_train)

# evaluate performance
print('Train set')
pred = rf.predict_proba(X_train_scaled)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = rf.predict_proba(X_test_scaled)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
""" Train set
Random Forests roc-auc: 0.9867917218059866
Test set
Random Forests roc-auc: 0.7312510370001659

- As expected, Random Forests shows no change in performance regardless of whether it is trained on a dataset with 
    scaled or unscaled features. 
- This model in particular, is over-fitting to the training set. So we need to do some work to remove the over-fitting. 
    That exceeds the scope of this example.
 """

# train adaboost on non-scaled features

# call the model
ada = AdaBoostClassifier(n_estimators=200, random_state=44)

# train the model
ada.fit(X_train, y_train)

# evaluate model performance
print('Train set')
pred = ada.predict_proba(X_train)
print('AdaBoost roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = ada.predict_proba(X_test)
print('AdaBoost roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
""" Train set
AdaBoost roc-auc: 0.7970629821021541
Test set
AdaBoost roc-auc: 0.7473867595818815 """

# train adaboost on scaled features

# call the model
ada = AdaBoostClassifier(n_estimators=200, random_state=44)

# train the model
ada.fit(X_train_scaled, y_train)

# evaluate model performance
print('Train set')
pred = ada.predict_proba(X_train_scaled)
print('AdaBoost roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = ada.predict_proba(X_test_scaled)
print('AdaBoost roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
"""
Train set
AdaBoost roc-auc: 0.7970629821021541
Test set
AdaBoost roc-auc: 0.7475250262706707

- As expected, AdaBoost shows no change in performance regardless of whether it is trained on a dataset with
    scaled or unscaled features

That is all for this example. I hope you enjoyed the information, and see you in the next one. """