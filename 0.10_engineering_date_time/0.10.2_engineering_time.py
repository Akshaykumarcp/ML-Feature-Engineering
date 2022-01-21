""" 
Engineering Time
- In this example, we are going to extract different ways of representing time from a timestamp. 
- We can extract for example:
        - hour
        - minute
        - second
        - data
        - elapsed time

We will create a toy dataset for the demonstration. """

import pandas as pd
import numpy as np
import datetime

# let's create a toy data set: 1 column 7 different timestamps,

# 1 hr difference between timestamp

date = pd.Series(pd.date_range('2015-1-5 11:20:00', periods=7, freq='H'))

df = pd.DataFrame(dict(date=date))

df
date
""" 0	2015-01-05 11:20:00
1	2015-01-05 12:20:00
2	2015-01-05 13:20:00
3	2015-01-05 14:20:00
4	2015-01-05 15:20:00
5	2015-01-05 16:20:00
6	2015-01-05 17:20:00 """

""" Extract the hr, minute and second """

df['hour'] = df['date'].dt.hour
df['min'] = df['date'].dt.minute
df['sec'] = df['date'].dt.second

df
""" 
                 date  hour  min  sec
0 2015-01-05 11:20:00    11   20    0
1 2015-01-05 12:20:00    12   20    0
2 2015-01-05 13:20:00    13   20    0
3 2015-01-05 14:20:00    14   20    0
4 2015-01-05 15:20:00    15   20    0
5 2015-01-05 16:20:00    16   20    0
6 2015-01-05 17:20:00    17   20    0 """

""" Extract time part """

df['time'] = df['date'].dt.time

df
""" 
                 date  hour  min  sec      time
0 2015-01-05 11:20:00    11   20    0  11:20:00
1 2015-01-05 12:20:00    12   20    0  12:20:00
2 2015-01-05 13:20:00    13   20    0  13:20:00
3 2015-01-05 14:20:00    14   20    0  14:20:00
4 2015-01-05 15:20:00    15   20    0  15:20:00
5 2015-01-05 16:20:00    16   20    0  16:20:00
6 2015-01-05 17:20:00    17   20    0  17:20:00 """

# Extract hr, min, sec, at the same time
# now let's repeat what we did in cell 3 in 1 command

df[['h','m','s']] = pd.DataFrame([(x.hour, x.minute, x.second) for x in df['time']])

df
""" 
                 date  hour  min  sec      time   h   m  s
0 2015-01-05 11:20:00    11   20    0  11:20:00  11  20  0
1 2015-01-05 12:20:00    12   20    0  12:20:00  12  20  0
2 2015-01-05 13:20:00    13   20    0  13:20:00  13  20  0
3 2015-01-05 14:20:00    14   20    0  14:20:00  14  20  0
4 2015-01-05 15:20:00    15   20    0  15:20:00  15  20  0
5 2015-01-05 16:20:00    16   20    0  16:20:00  16  20  0
6 2015-01-05 17:20:00    17   20    0  17:20:00  17  20  0 """

# Calculate time difference
# let's create another toy dataframe with 2 timestamp columns
# and 7 rows each, in the first column the timestamps change monthly,
# in the second column the timestamps change weekly

date1 = pd.Series(pd.date_range('2012-1-1 12:00:00', periods=7, freq='M'))
date2 = pd.Series(pd.date_range('2013-3-11 21:45:00', periods=7, freq='W'))
 
df = pd.DataFrame(dict(Start_date = date1, End_date = date2))

df
""" 
           Start_date            End_date
0 2012-01-31 12:00:00 2013-03-17 21:45:00
1 2012-02-29 12:00:00 2013-03-24 21:45:00
2 2012-03-31 12:00:00 2013-03-31 21:45:00
3 2012-04-30 12:00:00 2013-04-07 21:45:00
4 2012-05-31 12:00:00 2013-04-14 21:45:00
5 2012-06-30 12:00:00 2013-04-21 21:45:00
6 2012-07-31 12:00:00 2013-04-28 21:45:00 """

# let's calculate the time elapsed in seconds

df['diff_seconds'] = df['End_date'] - df['Start_date']
df['diff_seconds']=df['diff_seconds']/np.timedelta64(1,'s')

df
""" 
           Start_date            End_date  diff_seconds
0 2012-01-31 12:00:00 2013-03-17 21:45:00    35545500.0
1 2012-02-29 12:00:00 2013-03-24 21:45:00    33644700.0
2 2012-03-31 12:00:00 2013-03-31 21:45:00    31571100.0
3 2012-04-30 12:00:00 2013-04-07 21:45:00    29583900.0
4 2012-05-31 12:00:00 2013-04-14 21:45:00    27510300.0
5 2012-06-30 12:00:00 2013-04-21 21:45:00    25523100.0
6 2012-07-31 12:00:00 2013-04-28 21:45:00    23449500.0 """

# let's calculate the time elapsed in minutes
df['diff_seconds'] = df['End_date'] - df['Start_date']
df['diff_seconds']=df['diff_seconds']/np.timedelta64(1,'m')

df
""" 
           Start_date            End_date  diff_seconds
0 2012-01-31 12:00:00 2013-03-17 21:45:00      592425.0
1 2012-02-29 12:00:00 2013-03-24 21:45:00      560745.0
2 2012-03-31 12:00:00 2013-03-31 21:45:00      526185.0
3 2012-04-30 12:00:00 2013-04-07 21:45:00      493065.0
4 2012-05-31 12:00:00 2013-04-14 21:45:00      458505.0
5 2012-06-30 12:00:00 2013-04-21 21:45:00      425385.0
6 2012-07-31 12:00:00 2013-04-28 21:45:00      390825.0 """

""" For more details visit article: http://www.datasciencemadesimple.com/difference-two-timestamps-seconds-minutes-hours-pandas-python-2/

Work with different timezones
- below, we will see how to work with timestamps that are in different time zones.
 """
# first, let's create a toy dataframe with some timestamps in different time zones

df = pd.DataFrame()

df['time'] = pd.concat([
    pd.Series(
        pd.date_range(
            start='2014-08-01 09:00', freq='H', periods=3,
            tz='Europe/Berlin')),
    pd.Series(
        pd.date_range(
            start='2014-08-01 09:00', freq='H', periods=3, tz='US/Central'))
    ], axis=0)

df
""" 
                        time
0  2014-08-01 09:00:00+02:00
1  2014-08-01 10:00:00+02:00
2  2014-08-01 11:00:00+02:00
0  2014-08-01 09:00:00-05:00
1  2014-08-01 10:00:00-05:00
2  2014-08-01 11:00:00-05:00

- We can see the different timezones indicated by the +2 and -5, respect to the meridian.
 """

# to work with different time zones, first we unify the timezone to the central one
# setting utc = True

df['time_utc'] = pd.to_datetime(df['time'], utc=True)

# next we change all timestamps to the desired timezone, eg Europe/London
# in this example

df['time_london'] = df['time_utc'].dt.tz_convert('Europe/London')


df
"""
                         time	                time_utc	              time_london
0	2014-08-01 09:00:00+02:00	2014-08-01 07:00:00+00:00	2014-08-01 08:00:00+01:00
1	2014-08-01 10:00:00+02:00	2014-08-01 08:00:00+00:00	2014-08-01 09:00:00+01:00
2	2014-08-01 11:00:00+02:00	2014-08-01 09:00:00+00:00	2014-08-01 10:00:00+01:00
0	2014-08-01 09:00:00-05:00	2014-08-01 14:00:00+00:00	2014-08-01 15:00:00+01:00
1	2014-08-01 10:00:00-05:00	2014-08-01 15:00:00+00:00	2014-08-01 16:00:00+01:00
2	2014-08-01 11:00:00-05:00	2014-08-01 16:00:00+00:00	2014-08-01 17:00:00+01:00 """

""" 
- For feature engineering, we can of course, set all timezones to the central, utc=True, and work with that to 
        extract time elapsed etc. The additional timezone encoding is mostly for human readability.

I hope you enjoyed this example, and see you in the next one!
 """
