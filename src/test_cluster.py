import numpy as np
import pandas as pd

from cluster import Cluster
from item import Item
from range import Range

def test_generalise():

    headers = ["pid", "Age", "Salary"]

    a = Item(
        pd.Series(
            data=np.array([1, 25, 30]),
            index=headers
        ),
        headers[1:],
        sensitive_attr=None
    )

    b = Item(
        pd.Series(
            data=np.array([1, 22, 27]),
            index=headers
        ),
        headers[1:],
        sensitive_attr=None
    )

    c = Cluster(headers[1:])

    c.insert(a)
    c.insert(b)

    np.random.seed(42)
    t = c.generalise(a)[0]

    generalised_headers = [
        "minAge", "spcAge", "maxAge",
        "minSalary", "spcSalary", "maxSalary"
    ]

    expected = Item(
        pd.Series(
            data=np.array([22, 25, 25, 27, 27, 30]),
            index=generalised_headers
        ),
        generalised_headers,
        sensitive_attr=None
    )

    assert t == expected

    # We should get the same values a second time
    t = c.generalise(b)[0]
    assert t == expected

def test_within_bounds_after_insert():
    headers = ["Age", "Salary"]

    t = Item(
        pd.Series(
            data=np.array([25, 27]),
            index=headers
        ),
        headers,
        sensitive_attr=None
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
        headers,
        sensitive_attr=None
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
        headers,
        sensitive_attr=None
    )

    c = Cluster(headers)
    c.ranges = {
        "Age": Range(20, 24),
        "Salary": Range(27, 32)
    }

    assert not c.within_bounds(t)

def test_no_cluster_intersection():
    headers = ["pid", "Age", "Salary"]

    t = Item(
        pd.Series(
            data=np.array([1, 25, 27]),
            index=headers
        ),
        headers,
        sensitive_attr=None
    )

    c1 = Cluster(headers)
    c2 = Cluster(headers)

    c1.insert(t)

    # Ensure only c1 contains an item
    assert len(c1.contents) == 1 and len(c2.contents) == 0

    c2.insert(t)

    # Ensure t was removed from c1 after insertion
    assert len(c1.contents) == 0 and len(c2.contents) == 1
