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

Numerical Variables
- The values of a numerical variable are numbers. 

They can be further classified into:

- Discrete variables
- Continuous variables

Discrete Variable
- In a discrete variable, the values are whole numbers (counts). 
- For example, the number of items bought by a customer in a supermarket is discrete. 
- The customer can buy 1, 25, or 50 items, but not 3.7 items. It is always a round number. 

The following are examples of discrete variables:

- Number of active bank accounts of a borrower (1, 4, 7, ...)
- Number of pets in the family
- Number of children in the family

Continuous Variable
- A variable that may contain any value within a range is continuous.
- For example, the total amount paid by a customer in a supermarket is continuous. 
- The customer can pay, GBP 20.5, GBP 13.10, GBP 83.20 and so on. 

Other examples of continuous variables are:
- House price (in principle, it can take any value) (GBP 350000, 57000, 100000, ...)
- Time spent surfing a website (3.4 seconds, 5.10 seconds, ...)
- Total debt as percentage of total income in the last month (0.2, 0.001, 0, 0.75, ...)

=============================================================================

In this example: Peer to peer lending (Finance)

In this example, we will use data from the peer-to-peer finance company Lending Club to inspect discrete and continuous numerical variables

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# let's load the dataset with just a few columns and a few rows to speed the example

# Variable definitions:
#-------------------------
# loan_amnt: loan amount
# int_rate: interest rate
# annual_inc: annual income
# open_acc: open accounts (more on this later)
# loan_status: loan status(paid, defaulted, etc)
# open_il_12m: accounts opened in the last 12 months

use_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'open_acc', 'loan_status','open_il_12m']

# this dataset is very big. To speed things up for the demo
# I will randomly select 10,000 rows when I load the dataset
# so I upload just 10,000 rows from the full dataset

data = pd.read_csv('dataset/loan.csv/loan.csv', usecols=use_cols).sample(10000, random_state=44)  # set a seed for reproducibility

data.head()
"""         loan_amnt  int_rate  annual_inc loan_status  open_acc  open_il_12m
131079    23675.0     15.80     90000.0     Current       4.0          NaN
384935     7500.0     11.67    102000.0     Current      11.0          NaN
73585     11150.0      9.25     72000.0  Fully Paid       6.0          NaN
660814     8000.0      6.24    101400.0     Current      20.0          NaN
349757    34000.0     19.52     99000.0     Current      12.0          NaN """

""" Continuous Variables """
# let's look at the values of the variable loan_amnt
# this is the amount of money requested by the borrower in US dollars
# this variable is continuous, it can take in principle any value

data.loan_amnt.unique()
""" array([23675.,  7500., 11150.,  8000., 34000.,  8725., 30000., 10000.,
       10200., 18000., 24000., 12000., 19200., 35000.,  4800., 16800.,
       14725., 16000., 31500., 22500., 18800., 14100., 20000.,  5000.,
       10625., 13350., 15000.,  2900., 28000., 15950.,  1500.,  1000.,
       25000., 19500., 13000., 21000., 17650.,  9600., 14525.,  5600.,
        6400.,  6000., 14000., 11000., 22400., 16825.,  8200., 21600.,
       20675.,  2000.,  3950.,  4000.,  9500.,  8975., 13625.,  7000.,
        5500.,  9450., 10800.,  7200., 27700.,  4650.,  3600.,  6450.,
       15625.,  4825., 11200.,  4200., 28500.,  3375.,  5800.,  4425.,
       25875., 24800., 14400., 22000., 19750., 26500.,  3625., 17000.,
       21125.,  7525.,  3000.,  9250., 14675., 11700.,  7025., 19775.,
       17800., 10300., 13200., 32000., 15400., 10575.,  9000., 10075.,
       14600., 11550., 12500., 20975., 26000., 22250., 17100., 13375.,
       12125., 23900., 28800., 12800., 10750.,  3500.,  4850., 18550.,
       12250., 10500.,  4225.,  3925.,  6950., 33000., 23250.,  7225.,
        1600., 14175., 23000., 11400.,  8500.,  2800., 20375., 10475.,
       19275.,  8900., 11250., 17600., 21150.,  6700.,  9150., 11325.,
       27800.,  5950., 29700., 24500., 12100.,  8425., 19600.,  3700.,
        2500., 11975.,  8225.,  1975.,  7750., 19100., 29000.,  4750.,
        5400., 23200., 30625., 16550., 15425., 20125., 28650., 30225.,
        9800.,  5550., 11500., 21200., 19175., 12375., 23325., 18300.,
        2950., 16450.,  6675., 17625.,  9125.,  9550., 15475., 21100.,
       15850., 16700.,  3850.,  7100., 11625., 20500., 13300.,  9750.,
       22800., 27000., 26400., 10250., 29975., 29275.,  7050., 18225.,
       10975., 21825., 22950.,  3775., 15600., 18875., 12575., 12475.,
        5750., 32200., 14375.,  3975., 22050., 24925., 14900., 26575.,
       15500.,  5300.,  3300.,  1200.,  8750., 13675.,  3425., 27175.,
        7600.,  6500., 29850., 13150., 17875., 10450.,  7925.,  8800.,
       10775., 11900., 17575., 19900., 13775.,  4175.,  6725., 17975.,
       19000.,  3200.,  5175., 28100.,  7800.,  6800., 16425., 19150.,
       16500., 23450., 17850., 17925.,  4925.,  7250., 16875., 22825.,
        4550., 12600., 11850., 14750., 13450.,  1800., 22375.,  4300.,
       10100., 11175., 24375., 23875.,  4450., 15825., 29325.,  4125.,
       25600., 18200., 12525., 12450.,  5375., 13600.,  4250., 12400.,
        1400., 23300.,  6300.,  7400., 22200.,  5025., 22225., 17500.,
       16200., 32400.,  4900.,  5200.,  7900., 30750.,  3250., 14025.,
        3400., 11950., 24900., 13525., 14500., 13700., 27575., 24325.,
       11575., 11750.,  7300., 12700., 25750.,  4950., 28900., 33950.,
       11425., 20400.,  8400., 25975., 32500.,  3650.,  3350.,  7575.,
        7075., 28250.,  5075., 24700., 20050.,  4725.,  2750., 14325.,
        8600., 14425., 17275., 13500., 21850., 23975.,  7175., 12425.,
       10050.,  4500.,  4700., 11050.,  2400.,  8125., 18250., 11300.,
       17425.,  2100., 16575.,  8275., 24625., 24450., 24525., 30925.,
       10950., 32625.,  5450., 25725., 10675., 13425., 19675.,  3875.,
       10400., 20250.,  6050., 12175., 18900., 23950., 16150., 27675.,
       25075.,  6075., 19050., 11525., 11075., 12075.,  8100.,  8250.,
       22750.,  3325.,  6475., 20325., 21350., 15275., 10225.,  2375.,
       21700., 19125., 27875., 20700.,  9975., 33600., 16900., 10150.,
        8875., 13750., 11800., 14075.,  7125., 16750.,  6250., 18500.,
       10825., 26900.,  6150.,  8150., 12875.,  1950., 34550.,  5050.,
        4475.,  8700., 11450., 25200., 27500.,  8325., 22550., 16975.,
       23475.,  7350., 31050.,  9700., 23350., 21275., 13050., 10325.,
        8850.,  9200.,  9925., 16100., 13650.,  2875., 11600., 27550.,
       18025., 19575.,  9375., 14450., 27250., 32100., 23750., 17950.,
       19075., 19800.,  8950., 10600., 18400., 23400.,  8450.,  8025.,
       12925.,  2325., 31975.,  4675., 14975., 18825., 31300., 27200.,
        6225., 20200., 15775., 21400., 19700.,  2475.,  9900., 12675.,
       15250., 23500., 16950., 11375., 28200.,  4375., 24350., 18350.,
       13800.,  5100.,  8300., 10375., 10275., 15300., 23650., 20425.,
        8175.,  6425., 14125.,  6825.,  3550., 31450., 15225., 13475.,
       17325., 20550., 12725., 30600., 15875.,  2700.,  9025., 12975.,
       30500.,  5925.,  7375.,  3900.,  9400.,  3825.,  9175.,  6875.,
       31000.,  3100.,  2525.,  2200., 15700., 11025., 25225., 23275.,
       32950.,  5250.,  2250., 27050., 14825., 26600.,  7825., 18575.,
       23850.,  5225.,  6350., 19725., 32650., 32675., 14575., 23575.,
        2150.,  3225., 16850., 11100.,  9675., 20875., 29175., 13900.,
       29900.,  7775., 21250., 29100., 24975.,  4075., 19300., 13225.,
       10550., 12750., 11875., 20800., 31825., 12275.,  1675., 21475.,
        9875., 10700., 26850.,  8775., 18600.,  9950., 24600., 10850.,
       15975.,  5700., 20150., 22850., 23825.,  9325.,  2850., 29775.,
        9100., 32225., 33575., 10900., 22100.,  3050., 12625., 15750.,
       11225., 13925., 13975., 14150.,  9650., 24250., 16225., 31200.,
        3025., 15125.,  4400., 17400., 19350., 18700., 12150., 18075.,
        9050., 23625.,  7150., 11125.,  6200., 24200., 32125., 30150.,
       19025.,  9825.,  1900., 12950., 14800.,  2275., 22575., 28975.,
       16300., 25275.,  9850., 22475., 16375., 22125., 30100., 24650.,
       15100., 27600., 33875., 12775., 27325., 24475.,  1450., 18625.,
       17825., 25450., 16250., 26375., 20950.,  2975., 15900., 15725.,
       16475., 34475., 24175.,  6625.,  9225., 11350., 16400.,  5725.,
       24400., 16125., 22600.,  3275., 10025., 23600., 22150., 30350.,
       32350.,  8650., 10525., 28675., 21900., 19475.,  1875.,  7475.,
       15375., 34025.,  9625., 33675., 24725.,  5275., 34625., 15800.,
        2600., 21300.,  4600., 29400., 23800., 11475.,  5875.,  8550.,
       23075., 20175., 22650., 11275.,  4100., 18050.,  5625., 20775.,
       17050., 17700., 23525., 10725.,  3575., 12300., 25475., 17475.,
        3800., 19950., 18150., 29050., 18750., 30800., 20225., 33500.,
       13325., 17225., 25150., 28375., 16725., 34800., 15200., 15450.,
       19375., 17750., 17075.,  7625.,  7725.,  3125., 10875.,  5150.,
       12900.,  3450., 26925., 15350.,  4875., 10350.,  7325., 21750.,
       33100.,  7950., 18450., 26975.,  9350.,  5425.,  9300.,  1925.,
       13250.,  6850.,  9525., 33175., 11675.,  7700., 20100., 25700.,
       10425.,  8525., 13125., 22075., 25350., 12225., 18175., 30375.,
        4275.,  5975., 16675., 13850., 14050., 25325.,  5125.,  9275.,
       11725., 20925., 13075., 13575., 27350., 28625.,  1100.,  6275.,
       16600., 10175., 32875.,  6750.,  9775., 20650.,  5825., 28850.,
       16625.,  6575., 17775.,  5350., 23125.,  2725.,  4625.,  8050.,
       12050.,  7875., 18975.,  3075., 17125., 14275., 33425., 17350.,
       24550., 27450., 19550., 21575., 14300., 27725., 26650.,  2775.,
        6125., 27525., 34500., 19825.,  8625., 29500., 18725., 12825.,
       10650.,  5475.,  5525., 22700., 22425., 21775., 20300., 20750.,
       24575., 19425.,  9075., 25300., 12350., 28050., 27625.,  8475.,
       30075., 14475.,  6600., 15075.,  7550., 13400.,  5325., 26125.,
       27850., 17550., 14700., 22525., 13950., 15925., 12325., 24100.,
       21450.,  7275.,  7850., 27100.,  6925., 19400.,  8350., 21725.,
       21650., 14950.,  5900.,  4575.,  6025., 31700., 28775., 26800.,
        4325., 13825., 11825.,  3725.,  9725., 13725., 21050., 32450.,
       25500., 19875., 24825., 29450.,  7975., 16525.,  4025., 11775.,
       19650., 27425., 24750., 13100.,  6100., 15550., 20450., 22675.,
       27300., 19850.,  1550., 26275.,  6975., 20475., 29575., 32750.,
       24950., 18525., 26750., 32850., 23100., 20350.,  7650., 14225.,
       12850., 26475., 28425., 12200., 15650., 32050., 33750., 13875.,
       18650., 13025.]) """

# let's make a histogram to get familiar with the distribution of the variable

fig = data.loan_amnt.hist(bins=50)
fig.set_title('Loan Amount Requested')
fig.set_xlabel('Loan Amount')
fig.set_ylabel('Number of Loans')
plt.savefig('0.1_types_of_variable_or_features/0.1.1_histogram_continous_variable_loan_amnt.png')
plt.show()

""" 
OBSERVATION:
- The values of the variable vary across the entire range of loan amounts typically disbursed to borrowers. 
- This is characteristic of continuous variables.

Note: 
- The taller bars correspond to loan sizes of 10000, 15000, 20000, and 35000, indicating that there are more loans disbursed for those loan amount values. 
- Likely, these particular loan amount values are offered as a default in the Lending Club website. 
- Less frequent loan values, like 23,000 or 33,000 are typically requested by people who require a specific amount of money for a definite purpose.
"""

# let's do the same exercise for the variable interest rate,
# which is the interest charged by lending club to the borrowers

# this variable is also continuous, it can take in principle any value within the range

data.int_rate.unique()
""" array([15.8 , 11.67,  9.25,  6.24, 19.52, 15.88, 17.86, 16.55, 25.83,
        8.9 , 10.64,  7.9 , 16.99, 20.99, 10.99,  9.67, 11.44, 11.53,
       13.11, 13.33, 17.57, 19.99, 15.41,  6.17, 18.25, 16.2 ,  9.99,
       24.89, 12.69,  9.32, 19.05,  7.89, 11.55,  9.76,  8.18, 20.49,
        8.67,  9.17, 16.59, 15.61, 18.24, 14.91,  7.62, 13.99, 11.14,
       16.29, 14.99, 13.98, 13.53, 12.49, 14.64,  6.62,  5.93, 25.57,
       13.65,  5.32,  6.49, 15.22, 13.18, 11.49,  9.71, 14.33, 24.99,
       15.59, 18.99, 14.65, 13.67,  6.68, 12.59, 11.48,  6.03, 19.19,
       12.05, 13.57, 13.35, 12.85, 10.16,  8.49, 17.27,  7.26,  7.12,
       13.48,  8.39, 12.39, 12.99, 14.49, 11.26,  8.19, 10.15, 17.56,
       27.88, 19.97,  9.49, 13.66, 12.29, 21.48, 11.99, 11.22, 17.14,
       10.74, 13.23, 15.31, 12.12, 18.29, 18.75, 18.54, 24.5 ,  6.92,
       19.24, 10.49,  7.49, 17.76,  6.99, 22.99, 14.16, 13.68, 14.75,
       13.49, 10.  , 19.72,  7.69, 12.88,  6.  , 18.84, 15.33, 23.4 ,
       18.55, 14.09,  7.88, 16.78, 20.5 , 19.03, 24.08, 14.98, 22.15,
        6.89, 17.1 , 14.27, 17.99,  9.8 , 21.  , 15.27,  6.39, 15.99,
        5.99,  7.91, 21.99, 23.13, 13.05, 15.95, 19.22, 14.96, 14.3 ,
       21.49, 14.31, 20.3 , 14.35, 12.35, 12.42, 12.84, 20.8 , 18.3 ,
        7.14, 20.2 , 25.8 , 15.58, 18.2 , 16.49, 16.02, 13.44, 23.99,
       22.7 , 11.83,  7.29, 13.79, 10.95, 17.77,  8.38, 21.7 , 14.48,
       18.92,  8.88, 23.1 , 22.45, 26.06, 14.47, 23.43, 10.65, 18.49,
       21.67, 14.84,  9.45, 26.77, 19.47, 21.6 ,  5.42,  9.63, 13.92,
       16.24, 19.69,  6.54, 15.1 , 23.63, 23.7 ,  7.51, 15.23, 20.31,
       25.89, 19.79, 17.91, 14.59,  6.91, 17.49,  7.66, 17.88, 25.99,
       11.71,  5.79, 11.03, 23.26, 19.2 , 22.95, 16.7 , 16.77, 27.31,
       17.97, 12.53, 24.7 , 10.37, 12.61, 15.28, 24.83, 15.81, 16.32,
       10.36, 10.59, 23.76, 15.7 , 13.06,  6.97, 15.65, 13.55, 23.28,
       21.98,  8.32, 11.11, 15.57, 12.68, 12.92, 15.13,  7.43, 15.77,
       22.2 , 19.74, 12.87, 14.61, 14.22, 12.73, 23.5 , 19.29, 13.93,
       18.79,  7.74, 25.78, 19.91, 15.96, 10.38, 18.61, 15.83, 22.9 ,
       15.2 , 17.58, 21.59, 16.82,  8.6 , 22.4 , 17.43, 13.61, 18.67,
       10.75, 11.12,  6.76, 13.43,  8.  , 21.18,  9.96, 11.36, 11.89,
       21.15, 18.53, 13.22,  9.83,  8.94, 18.85, 12.18, 16.69, 13.85,
       13.8 , 15.62, 14.11, 16.45, 14.5 , 22.48, 10.25, 11.86, 13.47,
       22.94, 12.98, 14.54, 18.64, 11.34,  9.91, 14.26, 10.71, 10.78,
       17.93, 21.74, 19.04, 11.66, 20.62, 21.27, 14.17, 23.83,  8.07,
       14.74, 19.82, 22.47, 11.58, 17.74,  9.62, 16.89, 24.24, 14.72,
       15.37, 19.48, 10.62, 18.39, 20.9 , 14.42, 12.41, 10.39, 12.21,
       11.72, 14.79, 19.42, 18.62, 14.85, 16.4 , 16.  , 14.46]) """
       
# let's make a histogram to get familiar with the distribution of the variable

fig = data.int_rate.hist(bins=30)

fig.set_title('Interest Rate')
fig.set_xlabel('Interest Rate')
fig.set_ylabel('Number of Loans')
plt.savefig('0.1_types_of_variable_or_features/0.1.1_histogram_continous_variable_int_rate.png')
plt.show()

"""
OBSERVATIONS:
- We see that the values of the variable vary continuously across the variable range. 
- The values are the interest rate charged to borrowers.
 """

# Now, let's explore the income declared by the customers, that is, how much they earn yearly.
# this variable is also continuous

fig = data.annual_inc.hist(bins=100)

# for better visualisation, I display only specific
# range in the x-axis
fig.set_xlim(0, 400000)

# title and axis legends
fig.set_title("Customer's Annual Income")
fig.set_xlabel('Annual Income')
fig.set_ylabel('Number of Customers')
plt.savefig('0.1_types_of_variable_or_features/0.1.1_histogram_continous_variable_annual_inc.png')
plt.show()

""" 
OBSERVATIONS:
- The majority of salaries are concentrated towards values in the range 30-70k, with only a few customers earning higher salaries. 
- The values of the variable, vary continuously across the variable range, because this is a continuous variable.

Discrete Variables
- Let's explore the variable "Number of open credit lines in the borrower's credit file" (open_acc in the dataset). 
- This variable represents the total number of credit items (for example, credit cards, car loans, mortgages, etc) that is known for that borrower. 
- By definition it is a discrete variable, because a borrower can have 1 credit card, but not 3.5 credit cards.
 """

# let's inspect the values of the variable this is a discrete variable

data.open_acc.dropna().unique()
""" array([ 4., 11.,  6., 20., 12.,  5.,  8.,  9., 18., 16., 10., 13., 14.,
        7., 19.,  3., 15., 17., 26.,  2., 27., 22., 21., 25., 23., 29.,
       39., 24., 30., 31., 28., 37., 32., 48., 33., 34.,  1., 35., 36.,
       41., 45., 40., 42.]) """

# let's make an histogram to get familiar with the distribution of the variable

fig = data.open_acc.hist(bins=100)

# for better visualisation, I display only specific
# range in the x-axis
fig.set_xlim(0, 30)

# title and axis legends
fig.set_title('Number of open accounts')
fig.set_xlabel('Number of open accounts')
fig.set_ylabel('Number of Customers')
plt.savefig('0.1_types_of_variable_or_features/0.1.1_histogram_discrete_variable_open_acc.png')
plt.show()

""" 
OBSERVATIONS:
- Histograms of discrete variables have this typical broken shape, as not all the values within the variable range are present in the variable. 
- As I said, the customer can have 3 credit cards, but not 3,5 credit cards.

Let's look at another example of a discrete variable in this dataset:
- Number of installment accounts opened in past 12 months (open_il_12m in the dataset).
- Installment accounts are those that at the moment of acquiring them, there is a set period and amount of repayments agreed between the lender and borrower. 
- An example of this is a car loan, or a student loan. 
- The borrower knows that they will pay a fixed amount over a fixed period, for example 36 months.
"""

# let's inspect the variable values
data.open_il_12m.unique()
""" array([nan,  1.,  0.,  2.,  3.,  4.,  6.]) """

# let's make a histogram to get familiar with the distribution of the variable

fig = data.open_il_12m.hist(bins=50)
fig.set_title('Number of installment accounts opened in past 12 months')
fig.set_xlabel('Number of installment accounts opened in past 12 months')
fig.set_ylabel('Number of Borrowers')
plt.savefig('0.1_types_of_variable_or_features/0.1.1_histogram_discrete_variable_open_il_12m.png')
plt.show()

""" 
OBSERVATIONS:
- The majority of the borrowers have none or 1 installment account, with only a few borrowers having more than 2.

A variation of discrete variables: the binary variable
Binary variables, are discrete variables, that can take only 2 values, therefore binary.

Below, I will create an additional variable, called defaulted, to capture the number of loans that have defaulted. 
    A defaulted loan is a loan that a customer has failed to re-pay and the money is lost.

The variable takes the values 0 where the loans are OK and being re-paid regularly, or 1, when the borrower has confirmed that will not be able to re-pay the borrowed amount.
"""

# let's inspect the values of the variable loan status

data.loan_status.unique()
""" array(['Current', 'Fully Paid', 'Default', 'Charged Off',
       'Late (31-120 days)', 'Issued', 'In Grace Period',
       'Does not meet the credit policy. Status:Fully Paid',
       'Does not meet the credit policy. Status:Charged Off',
       'Late (16-30 days)'], dtype=object) """

# let's create one additional variable called "defaulted".
# This variable indicates if the loan has defaulted, which means,
# if the borrower failed to re-pay the loan, and the money is deemed lost.

data['defaulted'] = np.where(data.loan_status.isin(['Default']), 1, 0)

data.defaulted.mean()
#0.0017

# the new variable takes the value of 0
# if the loan is not defaulted

data.head()
"""         loan_amnt  int_rate  annual_inc loan_status  open_acc  open_il_12m  defaulted
131079    23675.0     15.80     90000.0     Current       4.0          NaN          0
384935     7500.0     11.67    102000.0     Current      11.0          NaN          0
73585     11150.0      9.25     72000.0  Fully Paid       6.0          NaN          0
660814     8000.0      6.24    101400.0     Current      20.0          NaN          0
349757    34000.0     19.52     99000.0     Current      12.0          NaN          0 """

# Compare the columns 'loan_status' and 'defaulted' to convince yourself of the outcome of the previous function

# the new variable takes the value 1 for loans that
# are defaulted

data[data.loan_status.isin(['Default'])].head()
"""         loan_amnt  int_rate  annual_inc loan_status  open_acc  open_il_12m  defaulted
444724    10200.0     25.83     37000.0     Default       8.0          NaN          1
241318    22000.0     20.99     44000.0     Default      12.0          NaN          1
345729    20000.0     11.67    140000.0     Default       9.0          NaN          1
738216    10000.0     13.99     53000.0     Default       4.0          NaN          1
282110    11950.0     17.57     40000.0     Default      14.0          NaN          1 """

# Compare the columns 'loan_status' and 'defaulted' to convince yourself of the outcome of the previous function

# A binary variable, can take 2 values. For example in
# the variable "defaulted" that we just created:
# either the loan is defaulted (1) or not (0)

data.defaulted.unique()
# array([0, 1], dtype=int64)

# let's make a histogram, although histograms for
# binary variables do not make a lot of sense

fig = data.defaulted.hist()
fig.set_xlim(0, 2)
fig.set_title('Defaulted accounts')
fig.set_xlabel('Defaulted')
fig.set_ylabel('Number of Loans')
plt.savefig('0.1_types_of_variable_or_features/0.1.1_histogram_discrete_binary_variable_defaulted.png')
plt.show()

"""
OBSERVATIONS: 
- As we can see, the variable shows only 2 values, 0 and 1, and the majority of the loans are OK.

That is all for this example. I hope you enjoyed the information, and see you in the next one. """