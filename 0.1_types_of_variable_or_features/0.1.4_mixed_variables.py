""" 
Mixed Variables
- Mixed variables are those which values contain both numbers and labels.

- Variables can be mixed for a variety of reasons. 
- For example, 
    - when credit agencies gather and store financial information of users, usually, the values of the variables 
            they store are numbers. However, in some cases the credit agencies cannot retrieve information for a 
            certain user for different reasons. What Credit Agencies do in these situations is to code each different 
            reason due to which they failed to retrieve information with a different code or 'label'. Like this, 
            they generate mixed type variables. These variables contain numbers when the value could be retrieved, or 
            labels otherwise.

- As an example, think of the variable 'number_of_open_accounts'. It can take any number, representing the number 
        of different financial accounts of the borrower. Sometimes, information may not be available for a certain 
        borrower, for a variety of reasons. Each reason will be coded by a different letter, 
        for example: 'A': couldn't identify the person, 
                    'B': no relevant data, 
                    'C': person seems not to have any open account.

- Another example of mixed type variables, is for example the variable missed_payment_status. 
        This variable indicates, whether a borrower has missed a (any) payment in their financial item. 
        For example, if the borrower has a credit card, this variable indicates whether they missed a monthly 
        payment on it. Therefore, this variable can take values of 0, 1, 2, 3 meaning that the customer has 
        missed 0-3 payments in their account. And it can also take the value D, if the customer defaulted on that 
        account.

- Typically, once the customer has missed 3 payments, the lender declares the item defaulted (D), 
        that is why this variable takes numerical values 0-3 and then D.

For this example, you will use a toy csv file called sample_s2.csv. 
 """
import pandas as pd

import matplotlib.pyplot as plt
# open_il_24m indicates:
# "Number of installment accounts opened in past 24 months".
# Installment accounts are those that, at the moment of acquiring them,
# there is a set period and amount of repayments agreed between the
# lender and borrower. An example of this is a car loan, or a student loan.
# the borrowers know that they are going to pay a fixed amount over a fixed period

data = pd.read_csv('0.1_types_of_variable_or_features/sample_s2.csv')
data.head()
"""     id open_il_24m
0  1077501           C
1  1077430           A
2  1077175           A
3  1076863           A
4  1075358           A """

data.shape
# (887379, 2)

# Fictitious meaning of the different letters / codes
# in the variable:
# 'A': couldn't identify the person
# 'B': no relevant data
# 'C': person seems not to have any account open

data.open_il_24m.unique()
""" array(['C', 'A', 'B', '0.0', '1.0', '2.0', '4.0', '3.0', '6.0', '5.0',
       '9.0', '7.0', '8.0', '13.0', '10.0', '19.0', '11.0', '12.0',
       '14.0', '15.0'], dtype=object) """

# Now, let's make a bar plot showing the different number of 
# borrowers for each of the values of the mixed variable

fig = data.open_il_24m.value_counts().plot.bar()
fig.set_title('Number of installment accounts open')
fig.set_ylabel('Number of borrowers')
plt.show()
""" 
This is how a mixed variable looks like!
That is all for this example. I hope you enjoyed the information, and see you in the next one.
"""