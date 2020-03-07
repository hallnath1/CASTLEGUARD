import numpy as np
import pandas as pd

from cluster import Cluster
from item import Item
from range import Range

def test_generalise():

    frame = pd.read_csv("data.csv")

    t = Item(frame.iloc[0], headers=["PickupLocationID", "TripDistance"])
    a = Item(frame.iloc[1], headers=["PickupLocationID", "TripDistance"])
    b = Item(frame.iloc[2], headers=["PickupLocationID", "TripDistance"])
    c = Cluster(headers=["PickupLocationID", "TripDistance"])
    c.insert(t)
    c.insert(a)
    c.insert(b)

    t, orig = c.generalise(t)
    d = {"pid":1, "minPickupLocationID":49, "maxPickupLocationID":264, "minTripDistance":.00, "maxTripDistance":0.86}
    df = pd.Series(data=d)
    test = Item(df ,headers=["minPickupLocationID", "maxPickupLocationID", "minTripDistance", "maxTripDistance"])

    assert t == test

def test_within_bounds_after_insert():
    headers = ["Age", "Salary"]

    t = Item(
        pd.Series(
            data=np.array([25, 27]),
            index=headers
        ),
        headers
    )

    c = Cluster(headers)
    c.insert(t)

    assert c.within_bounds(t)

def test_within_bounds():
    headers = ["Age", "Salary"]

    t = Item(
        pd.Series(
            data=np.array([25, 27]),
            index=headers
        ),
        headers
    )

    c = Cluster(headers)
    c.ranges = {
        "Age": Range(20, 25),
        "Salary": Range(27, 32)
    }

    assert c.within_bounds(t)

def test_out_of_bounds():
    headers = ["Age", "Salary"]

    t = Item(
        pd.Series(
            data=np.array([25, 27]),
            index=headers
        ),
        headers
    )

    c = Cluster(headers)
    c.ranges = {
        "Age": Range(20, 24),
        "Salary": Range(27, 32)
    }

    assert not c.within_bounds(t)
