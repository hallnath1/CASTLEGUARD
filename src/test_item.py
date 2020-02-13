from item import Item
import pandas as pd

def test_tuple_distance():
    s = {'col1' : [1], 'col2' : [2], 'col3' : [3]}
    sdf = pd.DataFrame(data=s)
    t = {'col1' : [4], 'col2' : [5], 'col3' : [6]}
    tdf = pd.DataFrame(data=t)

    item_1 = Item(data=sdf, headers=['col1', 'col2', 'col3'])
    item_2 = Item(data=tdf, headers=['col1', 'col2', 'col3'])

    assert item_1.tuple_distance(item_2) == 3
