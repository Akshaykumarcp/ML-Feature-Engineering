""" 
What is a Variable?
- A variable is any characteristic, number, or quantity that can be measured or counted. 
- They are called 'variables' because the value they take may vary, and it usually does. 

The following are examples of variables:

- Age (21, 35, 62, ...)
- Gender (male, female)
- Income (GBP 20000, GBP 35000, GBP 45000, ...)
- House price (GBP 350000, GBP 570000, ...)
- Country of birth (China, Russia, Costa Rica, ...)
- Eye colour (brown, green, blue, ...)
- Vehicle make (Ford, Volkswagen, ...)

Most variables in a data set can be classified into one of two major types:

- Numerical variables
- Categorical variables

===================================================================================

Categorical Variables
- The values of a categorical variable are selected from a group of categories, also called labels. 
- Examples are gender (male or female) and marital status (never married, married, divorced or widowed). 

Other examples of categorical variables include:

- Intended use of loan (debt-consolidation, car purchase, wedding expenses, ...)
- Mobile network provider (Vodafone, Orange, ...)
- Postcode

Categorical variables can be further categorised into:

- Ordinal Variables
- Nominal variables

================================================================

Ordinal Variable
- Ordinal variables are categorical variable in which the categories can be meaningfully ordered. 

For example:

- Student's grade in an exam (A, B, C or Fail).
- Days of the week, where Monday = 1 and Sunday = 7.
- Educational level, with the categories Elementary school, High school, College graduate and PhD ranked from 1 to 4.

Nominal Variable
- For nominal variables, there isn't an intrinsic order in the labels. 

For example, country of birth, with values Argentina, England, Germany, etc., is nominal. 

Other examples of nominal variables include:

- Car colour (blue, grey, silver, ...)
- Vehicle make (Citroen, Peugeot, ...)
- City (Manchester, London, Chester, ...)

There is nothing that indicates an intrinsic order of the labels, and in principle, they are all equal.

To be considered:
- Sometimes categorical variables are coded as numbers when the data are recorded (e.g. gender may be coded as 0 for males and 1 for females). 
    The variable is still categorical, despite the use of numbers.
- In a similar way, individuals in a survey may be coded with a number that uniquely identifies them (for example to avoid storing personal information for confidentiality). 
    This number is really a label, and the variable then categorical. 
    The number has no meaning other than making it possible to uniquely identify the observation (in this case the interviewed subject).

- Ideally, when we work with a dataset in a business scenario, the data will come with a dictionary that indicates if the numbers in the variables are to be considered as categories or if they are numerical. 
    And if the numbers are categories, the dictionary would explain what each value in the variable represents.

=============================================================================

In this example, we will use data from the peer-o-peer finance company Lending Club to inspect nominal categorical variables

"""

import pandas as pd
import matplotlib.pyplot as plt

# let's load the dataset with just a few columns and a few rows, to speed things up

# Variable definitions:
#-------------------------
# purpose: intended use of the loan
# loan_status: loan statues, defaulted, paid, etc
# home_ownership: whether the borrower owns or rents their property

use_cols = ['id', 'purpose', 'loan_status', 'home_ownership']

# this dataset is very big. To speed things up for the demo
# I will randomly select 10,000 rows when I load the dataset
# so I upload just 10,000 rows from the full dataset

data = pd.read_csv('dataset/loan.csv/loan.csv', usecols=use_cols).sample(
    10000, random_state=44)  # set a seed for reproducibility

data.head()
""" 
              id home_ownership loan_status             purpose
131079   5554979           RENT     Current  debt_consolidation
384935  16552323            OWN     Current                 car
73585    8185454           RENT  Fully Paid         credit_card
660814  56180166           RENT     Current  debt_consolidation
349757  16772363           RENT     Current         credit_card """

# let's inspect the variable home ownership,
# which indicates whether the borrowers own their home
# or if they are renting, among other things.

data.home_ownership.unique()
""" array(['RENT', 'OWN', 'MORTGAGE', 'OTHER'], dtype=object) """

# let's make a bar plot, with the number of loans for each category of home ownership
# the code below counts the number of observations (borrowers)
# within each category and then makes a bar plot

fig = data['home_ownership'].value_counts().plot.bar()
fig.set_title('Home Ownership')
fig.set_ylabel('Number of customers')
plt.savefig('0.1_types_of_variable_or_features/0.1.2_barPlot_categorical_variable_home_ownership.png')
plt.show()

"""
OBSERVATIONS:
- The majority of the borrowers either own their house on a mortgage or rent their property. 
- A few borrowers own their home completely. 
- The category 'Other' seems to be empty. 

To be completely sure, we could print the numbers as below:
 """

data['home_ownership'].value_counts()
""" 
MORTGAGE    4957
RENT        4055
OWN          986
OTHER          2
Name: home_ownership, dtype: int64 """

"""
OBSERVATION: 
- There are 2 borrowers that have other arrangements for their property. 
For example, they could live with their parents, or live in a hotel.
 """

# the "purpose" variable is another categorical variable
# that indicates how the borrowers intend to use the
# money they are borrowing, for example to improve their
# house, or to cancel previous debt.

data.purpose.unique()
""" array(['debt_consolidation', 'car', 'credit_card', 'small_business',
       'house', 'moving', 'other', 'home_improvement', 'medical',
       'major_purchase', 'vacation', 'educational', 'wedding',
       'renewable_energy'], dtype=object) """

"""        
- Debt consolidation means that the borrower would like a loan to cancel previous debts, car means that the borrower is borrowing the money to buy a car, and so on. 
- It gives an idea of the intended use of the loan.
 """

# let's make a bar plot with the number of borrowers
# within each category

# the code below counts the number of observations (borrowers)
# within each category and then makes a plot

fig = data['purpose'].value_counts().plot.bar()
fig.set_title('Loan Purpose')
fig.set_ylabel('Number of customers')
plt.savefig('0.1_types_of_variable_or_features/0.1.2_barPlot_categorical_variable_purpose.png')
plt.show()

"""
OBSERVATION: 
- The majority of the borrowers intend to use the loan for 'debt consolidation' or to repay their 'credit cards'. 
- This is quite common.
- What the borrowers intend to do is, to consolidate all the debt that they have on different financial items, in one 
    single debt, the new loan that they will take from Lending Club in this case. 
- This loan will usually provide an advantage to the borrower, either in the form of lower interest rates than a credit
     card, for example, or longer repayment period.
"""

# let's look at one additional categorical variable,
# "loan status", which represents the current status
# of the loan. This is whether the loan is still active
# and being repaid, or if it was defaulted,
# or if it was fully paid among other things.

data.loan_status.unique()
""" array(['Current', 'Fully Paid', 'Default', 'Charged Off',
       'Late (31-120 days)', 'Issued', 'In Grace Period',
       'Does not meet the credit policy. Status:Fully Paid',
       'Does not meet the credit policy. Status:Charged Off',
       'Late (16-30 days)'], dtype=object) """

# let's make a bar plot with the number of borrowers within each category

fig = data['loan_status'].value_counts().plot.bar()
fig.set_title('Status of the Loan')
fig.set_ylabel('Number of customers')
plt.savefig('0.1_types_of_variable_or_features/0.1.2_barPlot_categorical_variable_loan_status.png')
plt.show()

""" 
OBSERVATIONS:

We can see that the majority of the loans are active (current) and a big number have been 'Fully paid'. 

The remaining labels have the following meaning:
- Late (16-30 days): customer missed a payment
- Late (31-120 days): customer is behind in payments for more than a month
- Charged off: the company declared that they will not be able to recover the money for that loan ( money is typically lost)
- Issued: loan was granted but money not yet sent to borrower
- In Grace Period: window of time agreed with customer to wait for payment, usually, when customer is behind in their payments """

# finally, let's look at a variable that is numerical,
# but its numbers have no real meaning
# their values are more "labels" than real numbers

data['id'].head()
""" 
131079     5554979
384935    16552323
73585      8185454
660814    56180166
349757    16772363
Name: id, dtype: int64 """

"""
- Each id represents one customer. 
- This number is assigned to identify the customer if needed, while maintaining confidentiality and ensuring data protection.
"""

# The variable has as many different id values as customers, in this case 10000, 
# remember that because we loaded only 
# 10000 rows/customers from the original dataset.

len(data['id'].unique())
# 10000

# That is all for this example. I hope you enjoyed the information, and see you in the next one.