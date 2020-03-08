from item import Item
import pandas as pd

def test_realistic_tuple_distance():
    frame = pd.read_csv("data.csv")

    a = Item(frame.iloc[0], headers=["PickupLocationID", "TripDistance"], sensitive_attr=None)
    b = Item(frame.iloc[1], headers=["PickupLocationID", "TripDistance"], sensitive_attr=None)

    assert a.tuple_distance(b) == 118.0883982447048
