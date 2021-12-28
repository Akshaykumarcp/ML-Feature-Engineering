""" 
Dates and Times
- A special type of categorical variable are those that instead of taking traditional labels, like color (blue, red),
     or city (London, Manchester), take dates and / or time as values. 
- For example, date of birth ('29-08-1987', '12-01-2012'), or date of application ('2016-Dec', '2013-March').

- Datetime variables can contain dates only, time only, or date and time.

We don't usually work with a datetime variable in their raw format because:
- Date variables contain a huge number of different categories
- We can extract much more information from datetime variables by preprocessing them correctly

In addition, 
- often, date variables will contain dates that were not present in the dataset used to train the machine learning model. 
- In fact, date variables will usually contain dates placed in the future, respect to the dates in the training dataset. 
- Therefore, the machine learning model will not know what to do with them, because it never saw them while being trained.

=============================================================================

In this example, we will use data from the peer-o-peer finance company Lending Club to inspect nominal categorical variables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# let's load the Lending Club dataset with a few selected columns

# Variable definitions:
#-------------------------
# loan_amnt: loan amount requested by borrower
# grade: risk markets in which borrowers are placed
# purpose: intended use of the loan
# issue_d: date the loan was issued
# last_pymnt_d: date of last payment towards repyaing the loan

use_cols = ['loan_amnt', 'grade', 'purpose', 'issue_d', 'last_pymnt_d']

data = pd.read_csv('dataset/loan.csv/loan.csv', usecols=use_cols)

data.head()
"""    
    loan_amnt grade   issue_d         purpose last_pymnt_d
0     5000.0     B  Dec-2011     credit_card     Jan-2015
1     2500.0     C  Dec-2011             car     Apr-2013
2     2400.0     C  Dec-2011  small_business     Jun-2014
3    10000.0     C  Dec-2011           other     Jan-2015
4     3000.0     B  Dec-2011           other     Jan-2016 """

# pandas assigns type 'object' when reading dates and considers them strings.
# Let's have a look

data.dtypes
""" 
loan_amnt       float64
grade            object
issue_d          object
purpose          object
last_pymnt_d     object
dtype: object """

"""
OBSERVATIONS:
- Both issue_d and last_pymnt_d are casted as objects. Therefore, pandas will treat them as strings or categorical variables.
- In order to instruct pandas to treat them as dates, we need to re-cast them into datetime format. See below.
 """

# now let's parse the dates, currently coded as strings, into datetime format
# this will allow us to make some analysis afterwards

data['issue_dt'] = pd.to_datetime(data.issue_d)
data['last_pymnt_dt'] = pd.to_datetime(data.last_pymnt_d)

data[['issue_d', 'issue_dt', 'last_pymnt_d', 'last_pymnt_dt']].head()
""" 
issue_d	        issue_dt	last_pymnt_d	last_pymnt_dt
0	Dec-2011	2011-12-01	Jan-2015	2015-01-01
1	Dec-2011	2011-12-01	Apr-2013	2013-04-01
2	Dec-2011	2011-12-01	Jun-2014	2014-06-01
3	Dec-2011	2011-12-01	Jan-2015	2015-01-01
4	Dec-2011	2011-12-01	Jan-2016	2016-01-01 """

# let's see how much money Lending Club has disbursed
# (i.e., lent) over the years to the different risk
# markets (grade variable)

fig = data.groupby(['issue_dt', 'grade'])['loan_amnt'].sum().unstack().plot(
    figsize=(14, 8), linewidth=2)

fig.set_title('Disbursed amount in time')
fig.set_ylabel('Disbursed Amount (US Dollars)')
plt.savefig('0.1_types_of_variable_or_features/0.1.3_Plot_datetime_variable_disbursed_amount.png')
plt.show()

""" 
OBSERVATIONS:
- Lending Club seems to have increased the amount of money lent from 2013 onwards. 
- The tendency indicates that they continue to grow. 
- In addition, we can see that their major business comes from lending money to C and B grades.

- 'A' grades are the lower risk borrowers, borrowers that most likely will be able to repay their loans, as they are 
    typically in a better financial situation. Borrowers within this grade are charged lower interest rates.

- E, F and G grades represent the riskier borrowers. Usually borrowers in somewhat tighter financial situations, or 
    for whom there is not sufficient financial history to make a reliable credit assessment. 
    They are typically charged higher rates, as the business, and therefore the investors, take a higher risk when 
    lending them money.

That is all for this example. I hope you enjoyed the information, and see you in the next one. """