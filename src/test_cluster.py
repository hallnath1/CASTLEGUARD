from cluster import Cluster
from item import Item
import pandas as pd

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
