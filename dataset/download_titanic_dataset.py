import pandas as pd
import numpy as np

# load dataset from openML
data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')

# replace ? with np.nan
data = data.replace('?', np.nan)

# capture only first cabin when more than one is available
def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan

data['cabin'] = data['cabin'].apply(get_first_cabin)

# save to csv
data.to_csv('titanic.csv', index=False)