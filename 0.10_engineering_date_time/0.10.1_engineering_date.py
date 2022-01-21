""" 
Engineering Dates
- Date variables are special type of categorical variable. 
- By their own nature, date variables will contain a multitude of different labels, each one corresponding to a 
              specific date and sometimes time. 
- Date variables, when preprocessed properly can highly enrich a dataset. 
- For example, from a date variable we can extract:
       - Week of the year
       - Month
       - Quarter
       - Semester
       - Year
       - Day (number)
       - Day of the week
       - Is Weekend?
       - Time differences in years, months, days, hrs, etc.

- Date variables should not be used as categorical variables when building a machine learning model. 
- Not only because they have a multitude of categories, but also because when we actually use the model to score a 
              new observation, this observation will most likely be in the future, an therefore its date label, 
              might be different from the ones contained in the training set and therefore the ones used to train 
              the machine learning algorithm.

In this example: 
In this example, we will use data from the peer-o-peer finance company Lending Club to extract different features 
              from datetime variables.
"""

import pandas as pd
import numpy as np
import datetime

# let's load the Lending Club dataset with a few selected columns
# and just a few rows to speed things up

use_cols = ['issue_d', 'last_pymnt_d']
data = pd.read_csv('dataset/loan.csv/loan.csv', usecols=use_cols, nrows=10000)

data.head()
""" 
    issue_d last_pymnt_d
0  Dec-2011     Jan-2015
1  Dec-2011     Apr-2013
2  Dec-2011     Jun-2014
3  Dec-2011     Jan-2015
4  Dec-2011     Jan-2016 """

# now let's parse the dates, currently cast as strings, into datetime format
data['issue_dt'] = pd.to_datetime(data.issue_d)
data['last_pymnt_dt'] = pd.to_datetime(data.last_pymnt_d)

data[['issue_d','issue_dt','last_pymnt_d', 'last_pymnt_dt']].head()
""" 
    issue_d   issue_dt last_pymnt_d last_pymnt_dt
0  Dec-2011 2011-12-01     Jan-2015    2015-01-01
1  Dec-2011 2011-12-01     Apr-2013    2013-04-01
2  Dec-2011 2011-12-01     Jun-2014    2014-06-01
3  Dec-2011 2011-12-01     Jan-2015    2015-01-01
4  Dec-2011 2011-12-01     Jan-2016    2016-01-01 """

# Extract week of the year
# Extracting week of year from date, varies from 1 to 52

data['issue_dt_week'] = data['issue_dt'].dt.week

data[['issue_dt', 'issue_dt_week']].head()
""" 
    issue_dt  issue_dt_week
0 2011-12-01             48
1 2011-12-01             48
2 2011-12-01             48
3 2011-12-01             48
4 2011-12-01             48 """

data['issue_dt_week'].unique()
# array([48, 44, 39, 35, 31], dtype=int64)

""" Extract month """

# Extracting month from date - 1 to 12

data['issue_dt_month'] = data['issue_dt'].dt.month

data[['issue_dt', 'issue_dt_month']].head()
""" 
    issue_dt  issue_dt_month
0 2011-12-01              12
1 2011-12-01              12
2 2011-12-01              12
3 2011-12-01              12
4 2011-12-01              12 """

data['issue_dt_month'].unique()
# array([12, 11, 10,  9,  8], dtype=int64)

""" Extract quarter """
# Extract quarter from date variable - 1 to 4

data['issue_dt_quarter'] = data['issue_dt'].dt.quarter

data[['issue_dt', 'issue_dt_quarter']].head()
"""    issue_dt	issue_dt_quarter
0	2011-12-01	4
1	2011-12-01	4
2	2011-12-01	4
3	2011-12-01	4
4	2011-12-01	4
 """
 
data['issue_dt_quarter'].unique()
#  array([4, 3], dtype=int64)

""" Extract semester """
# We could also extract semester

data['issue_dt_semester'] = np.where(data['issue_dt_quarter'].isin([1,2]), 1, 2)

data.head()
""" 
    issue_d last_pymnt_d   issue_dt last_pymnt_dt  issue_dt_week  issue_dt_month  issue_dt_quarter  issue_dt_semester
0  Dec-2011     Jan-2015 2011-12-01    2015-01-01             48              12                 4                  2
1  Dec-2011     Apr-2013 2011-12-01    2013-04-01             48              12                 4                  2
2  Dec-2011     Jun-2014 2011-12-01    2014-06-01             48              12                 4                  2
3  Dec-2011     Jan-2015 2011-12-01    2015-01-01             48              12                 4                  2
4  Dec-2011     Jan-2016 2011-12-01    2016-01-01             48              12                 4                  2 """

data['issue_dt_semester'].unique()
# array([2], dtype=int64)

""" Extract year """
# extract year 

data['issue_dt_year'] = data['issue_dt'].dt.year

data[['issue_dt', 'issue_dt_year']].head()
""" 
       issue_dt	issue_dt_year
0	2011-12-01	2011
1	2011-12-01	2011
2	2011-12-01	2011
3	2011-12-01	2011
4	2011-12-01	2011 """

data['issue_dt_year'].unique()
# array([2011], dtype=int64)

""" Extract days, in various formats """
# day - numeric from 1-31

data['issue_dt_day'] = data['issue_dt'].dt.day

data[['issue_dt', 'issue_dt_day']].head()
"""
        issue_dt	issue_dt_day
0	2011-12-01	1
1	2011-12-01	1
2	2011-12-01	1
3	2011-12-01	1
4	2011-12-01	1 """

data['issue_dt_day'].unique()
# array([1], dtype=int64)

# day of the week - from 0 to 6

data['issue_dt_dayofweek'] = data['issue_dt'].dt.dayofweek

data[['issue_dt', 'issue_dt_dayofweek']].head()
""" 
       issue_dt	issue_dt_dayofweek
0	2011-12-01	3
1	2011-12-01	3
2	2011-12-01	3
3	2011-12-01	3
4	2011-12-01	3 """

data['issue_dt_dayofweek'].unique()
# array([3, 1, 5, 0], dtype=int64)

# day of the week - name

data['issue_dt_dayofweek'] = data['issue_dt'].dt.weekday_name

data[['issue_dt', 'issue_dt_dayofweek']].head()
"""    issue_dt	issue_dt_dayofweek
0	2011-12-01	Thursday
1	2011-12-01	Thursday
2	2011-12-01	Thursday
3	2011-12-01	Thursday
4	2011-12-01	Thursday
 """

data['issue_dt_dayofweek'].unique()
# array(['Thursday', 'Tuesday', 'Saturday', 'Monday'], dtype=object)
# was the application done on the weekend?

data['issue_dt_is_weekend'] = np.where(data['issue_dt_dayofweek'].isin(['Sunday', 'Saturday']), 1,0)

data[['issue_dt', 'issue_dt_dayofweek','issue_dt_is_weekend']].head()
"""    issue_dt	issue_dt_dayofweek	issue_dt_is_weekend
0	2011-12-01	Thursday	0
1	2011-12-01	Thursday	0
2	2011-12-01	Thursday	0
3	2011-12-01	Thursday	0
4	2011-12-01	Thursday	0
 """

data['issue_dt_is_weekend'].unique()
# array([0, 1], dtype=int64)

""" Extract Time elapsed between dates """
# perhaps more interestingly, extract the date difference between 2 dates

data['issue_dt'] - data['last_pymnt_dt']
""" 0      -1127 days
1       -487 days
2       -913 days
3      -1127 days
4      -1492 days
5      -1127 days
6      -1492 days
7      -1127 days
8       -122 days
9       -336 days
10      -548 days
11      -640 days
12      -213 days
13     -1127 days
14      -670 days
15     -1127 days
16     -1127 days
17      -517 days
18     -1158 days
19      -213 days
20     -1339 days
21      -640 days
22     -1127 days
23      -670 days
24      -305 days
25      -275 days
26            NaT
27      -366 days
28      -487 days
29     -1096 days
          ...    
9970   -1096 days
9971   -1188 days
9972    -519 days
9973    -274 days
9974   -1096 days
9975   -1004 days
9976    -274 days
9977   -1096 days
9978   -1096 days
9979    -762 days
9980   -1492 days
9981    -397 days
9982   -1004 days
9983     -31 days
9984    -974 days
9985   -1096 days
9986   -1096 days
9987    -731 days
9988    -853 days
9989    -915 days
9990    -244 days
9991   -1096 days
9992   -1096 days
9993    -458 days
9994   -1614 days
9995    -974 days
9996   -1522 days
9997   -1096 days
9998   -1096 days
9999    -153 days
Length: 10000, dtype: timedelta64[ns] """

# same as above capturing just the time difference

(data['last_pymnt_dt'] - data['issue_dt']).dt.days.head()
""" 0    1127.0
1     487.0
2     913.0
3    1127.0
4    1492.0
dtype: float64 """

# calculate number of months passed between 2 dates

data['months_passed'] = (data['last_pymnt_dt'] - data['issue_dt']) / np.timedelta64(1, 'M')
data['months_passed'] = np.round(data['months_passed'],0)

data[['last_pymnt_dt', 'issue_dt','months_passed']].head()
""" last_pymnt_dt	issue_dt	months_passed
0	2015-01-01	2011-12-01	37.0
1	2013-04-01	2011-12-01	16.0
2	2014-06-01	2011-12-01	30.0
3	2015-01-01	2011-12-01	37.0
4	2016-01-01	2011-12-01	49.0 """

# or the time difference to today

(datetime.datetime.today() - data['issue_dt']).head()
""" 0   2854 days 15:27:20.297171
1   2854 days 15:27:20.297171
2   2854 days 15:27:20.297171
3   2854 days 15:27:20.297171
4   2854 days 15:27:20.297171
Name: issue_dt, dtype: timedelta64[ns] """

(datetime.datetime.today() - data['issue_dt']).unique()
""" array([246641263284274000, 249233263284274000, 251911663284274000,
       254503663284274000, 257182063284274000], dtype='timedelta64[ns]') 
       
I hope you enjoyed this example, and see you in the next one!"""