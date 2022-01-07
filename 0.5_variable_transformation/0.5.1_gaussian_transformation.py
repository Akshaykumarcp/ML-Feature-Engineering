""" 
Gaussian Transformation
- Some machine learning models like linear and logistic regression assume that the variables are normally distributed.
        Often, variables are not normally distributed, but, transforming the variables to map their distribution to 
        a Gaussian distribution may, and often does, boost the performance of the machine learning algorithm.

- If a variable is not normally distributed, it is often possible to find a mathematical transformation to 
        normalise its distribution.

How can we transform variables so that they follow a normal distribution?
The most commonly used methods to transform variables are:
- Logarithmic transformation - np.log(X)
- Reciprocal transformation - 1 / X
- Square root transformation - X**(1/2)
- Exponential transformation (more general, you can use any exponent)
- Box-Cox transformation
- Yeo-Johnson transformation

You can find the formulas for Box-Cox and Yeo-Johnson here https://scikit-learn.org/stable/modules/preprocessing.html#mapping-to-a-gaussian-distribution

Briefly, 
- the Box-Cox transformation is an adaptation of the exponential transformation, scanning through various 
        exponents, and it already represents the untransformed variable, as well as the log transformed, 
        reciprocal, square and cube root transformed, as the lambda varies across the range of -5 to 5 
        (see formula, to understand this better). 
        So by doing Box-Cox transformation, in a way, we are evaluating all the other transformations and 
        choosing the best one. Box-Cox can only be applied to positive variables.
- Yeo-Johnson is a modification of the Box-Cox transformation so that it can be applied as well to non-positive 
        variables

To read more about the Box-Cox transformation visit any of these links:
- http://www.statisticshowto.com/box-cox-transformation/
- http://www.itl.nist.gov/div898/handbook/eda/section3/eda336.htm
- http://onlinestatbook.com/2/transformations/box-cox.html

To read more about the Yeo-Johnson transformation follow this link: https://www.stat.umn.edu/arc/yjpower.pdf

In this example:
- We will see how to implement variable transformations using NumPy and the House Prices dataset.
 """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# load the data
data = pd.read_csv('dataset/house-prices-advanced-regression-techniques/train.csv')

data.head()
""" 
   Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape LandContour Utilities  ... PoolArea PoolQC Fence MiscFeature MiscVal MoSold YrSold  SaleType  SaleCondition  SalePrice
0   1          60       RL         65.0     8450   Pave   NaN      Reg         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   2008        WD         Normal     208500
1   2          20       RL         80.0     9600   Pave   NaN      Reg         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      5   2007        WD         Normal     181500
2   3          60       RL         68.0    11250   Pave   NaN      IR1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      9   2008        WD         Normal     223500
3   4          70       RL         60.0     9550   Pave   NaN      IR1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   2006        WD        Abnorml     140000
4   5          60       RL         84.0    14260   Pave   NaN      IR1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0     12   2008        WD         Normal     250000 """

# let's explore the distribution of the numerical variables
data.hist(figsize=(20,20))
plt.show()

""" 
Plots to assess normality
- To visualise the distribution of the variables, we plot a histogram and a Q-Q plot. 
- In the Q-Q pLots, if the variable is normally distributed, the values of the variable should fall in a 45 degree 
        line when plotted against the theoretical quantiles.
 """

# plot the histograms to have a quick look at the variable distribution
# histogram and Q-Q plots

def diagnostic_plots(df, variable):
    
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable
    
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist(bins=30)

    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)

    plt.show()

# Original distribution
diagnostic_plots(data, 'GrLivArea')

""" We can see in the plots that the variable is not normally distributed. 
The values depart from the red line towards the ends of the distribution and we can see in the histogram that
 it is skewed to the right.
 """

# Logarithmic transformation
### Logarithmic transformation

data['GrLivArea_log'] = np.log(data['GrLivArea'])

diagnostic_plots(data, 'GrLivArea_log')

# The logarithmic transformation returned a more Gaussian for this variable.

# Reciprocal transformation
data['GrLivArea_reciprocal'] = 1 / (data['GrLivArea']) 

# np.reciprocal(data['GrLivArea'])

diagnostic_plots(data, 'GrLivArea_reciprocal')

# The reciprocal transformation was not useful to transform this variable.

# Square root transformation
data['GrLivArea_sqr'] = data['GrLivArea']**(1/2) 

# np.power(data['GrLivArea'], 1/2), np.sqrt(data['GrLivArea'])

diagnostic_plots(data, 'GrLivArea_sqr')

# The square root transformation offers a good alternative to normalise this variable.

# Exponential transformation
data['GrLivArea_exp'] = data['GrLivArea']**(1/1.5) # you can vary the exponent as needed

# np.power(data['GrLivArea'], any exponent we want)

diagnostic_plots(data, 'GrLivArea_exp')

# The exponential transformation did not work so nicely for this variable.

# Box-Cox transformation
""" 
The Box-Cox transformation is defined as:
- T(Y)=(Y exp(λ)−1)/λ if λ!=0, or log(Y) otherwise.
        - where Y is the response variable and λ is the transformation parameter. 
            λ varies from -5 to 5. 
            In the transformation, all values of λ are considered and the optimal value for a given variable
            is selected.

- Briefly, for each λ (the transformation tests several λs), the correlation coefficient of the Probability Plot 
        (Q-Q plot below, correlation between ordered values and theoretical quantiles) is calculated 
        (this optimisation equation actually varies with the implementation).
- The value of λ corresponding to the maximum correlation on the plot is then the optimal choice for λ.

In python, we can evaluate and obtain the best λ with the stats.boxcox function from the package scipy.
Let's have a look. """

data['GrLivArea_boxcox'], param = stats.boxcox(data['GrLivArea']) 

print('Optimal λ: ', param)
# Optimal λ:  0.006304886038020439

diagnostic_plots(data, 'GrLivArea_boxcox')

""" 
Yeo-Johnson transformation
- Yeo-Johnson is the same as Box-Cox for the positive values of the variable, but it has different equations 
        for the negative values of the variable as described here

- Again, the function searches over a bunch of λ and selects the one that returns the best fit to a normal distribution.
 """

# to avoid a NumPy error
data['GrLivArea'] = data['GrLivArea'].astype('float')
data['GrLivArea_yeojohnson'], param = stats.yeojohnson(data['GrLivArea']) 

print('Optimal λ: ', param)
# Optimal λ:  0.005539641278705955

diagnostic_plots(data, 'GrLivArea_yeojohnson')

""" 
- We can see that after the transformation, the quantiles are somewhat more aligned over the 45 degree 
        line with the theorical quantiles of the Gaussian distribution. """

