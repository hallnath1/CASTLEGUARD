import pandas as pd

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

def average_series(series_obj):
    """
    For the object, inspect its keys for 
    min/max entries and add them to a set. 
    For each item in that set, average the 
    min/max entries and return a dataframe 
    row with this information.
    """
    return series_obj

def average_group(group):
    """
    Go through the list of SERIES object in the 
    group. For each SERIES object, check the keys 
    for min/max entries. If there is one, add it 
    to the set of keys with a min/max entry. Then 
    go over these entries and average each min/max 
    and create a dataframe with all the averaged 
    data in it and return it. 
    """
    for s in group:
        print("SERIES IN GROUP: \n{}".format(average_series(s)))
    return group