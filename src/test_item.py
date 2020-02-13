from item import Item
import pandas as pd

def test_realistic_tuple_distance():
    frame = pd.read_csv("data.csv")

    a = Item(frame.iloc[0], headers=["PickupLocationID", "TripDistance"])
    b = Item(frame.iloc[1], headers=["PickupLocationID", "TripDistance"])

    assert a.tuple_distance(b) == 118.0883982447048
