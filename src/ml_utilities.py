import pandas as pd
import re
import numpy as np
from typing import List, Dict, Tuple

"""
This module will contain methods useful for ML
stuff in CASTLEGUARD.
An example of a method which will be in this
module is one to take a Series object and turns
it into a dataframe. Within the series object,
if any parts have been generalised, the min and
max will be averaged and that result will be used
instead.
"""

def average_series(series_obj: pd.Series) -> pd.Series:
    """
    For the object, inspect its keys for
    min/max entries and add them to a set.
    For each item in that set, average the
    min/max entries and return a dataframe
    row with this information.
    """
    labels = series_obj.keys()
    data = []
    axes = []
    values = {}
    for key in labels:
        m = re.search("((?<=min)|(?<=spc)|(?<=max))[A-Za-z]*",key)
        if m and m.group(0):
            if m.group(0) in values and len(values[m.group(0)]) == 2:
                total = (sum(values[m.group(0)]) + series_obj.get(key))/3
                if isinstance(values[m.group(0)][0], np.int64):
                    total = int(total)
                data.append(total)
                axes.append(m.group(0))
                del values[m.group(0)]
            elif m.group(0) in values:
                values[m.group(0)].append(series_obj.get(key))
            else:
                values[m.group(0)] = [series_obj.get(key)]
        else:
            data.append(series_obj.get(key))
            axes.append(key)
    return pd.Series(data, axes)

def average_group(group: List[pd.Series], datatypes:List[Tuple]=[]) -> pd.DataFrame:
    """
    Go through the list of SERIES object in the
    group. For each SERIES object, check the keys
    for min/max entries. If there is one, add it
    to the set of keys with a min/max entry. Then
    go over these entries and average each min/max
    and create a dataframe with all the averaged
    data in it and return it.
    """
    dataframes = []
    for s in group:
        avg = average_series(s)
        df = avg.to_frame().transpose()
        dataframes.append(df)

    whole = pd.concat(dataframes, ignore_index=True, sort=True) 
    for t in datatypes:
        whole[t[0]] = whole[t[0]].astype(t[1])
    
    return whole

def group_data(group: List[pd.Series], datatypes:List[Tuple]=[]) -> pd.DataFrame:
    dataframes = []
    for s in group:
        df = s.to_frame().transpose()
        dataframes.append(df)

    whole = pd.concat(dataframes, ignore_index=True, sort=True) 
    for t in datatypes:
        whole[t[0]] = whole[t[0]].astype(t[1])
    
    return whole

def process(data: pd.DataFrame, category: Dict) -> pd.DataFrame :
    data_keys = list(data.keys())
    cat_keys = list(category.keys())
    series_array = []
    for(_, row) in data.iterrows():
        values = []
        for cat in data_keys:
            if cat in cat_keys:
                values.append(category[cat].index(row[cat]))
            else:
                values.append(row[cat])
        s_val = pd.Series(values, data_keys)
        series_array.append(s_val.to_frame().transpose())
    return pd.concat(series_array, ignore_index=True)