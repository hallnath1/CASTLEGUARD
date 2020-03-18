"""Contains the Item class for manipulating tuples in CASTLE, typically meaning
a single row from a dataset."""

import math

from typing import Any, List

import pandas as pd

class Item:

    """A singular tuple within the CASTLE algorithm. Provides operations
    between tuples such as the distance, and allows tracking tuples more
    easily."""

    def __init__(self, data: pd.Series, headers: List[str], sensitive_attr: str):
        """Initialises an Item object

        Args:
            data: The data that the item contains
            headers: The columns/headers that we care about anonymising

        """
        self.data: pd.Series = data
        self.headers: List[str] = headers
        self.sensitive_attr: float = data[sensitive_attr] if sensitive_attr else None
        self.parent = None

    def tuple_distance(self, other) -> float:
        """Calculates the distance between the two tuples

        Args:
            other: The tuple to calculate the distance to

        Returns: The distance to the tuple

        """
        error = self.data[self.headers].sub(other.data[self.headers]).abs()
        mean_squared_error = error.pow(2).mean(axis=0)
        return math.sqrt(mean_squared_error)

    def update_attribute(self, header: str, value: float):
        """Updates a value in the tuple's data

        Args:
            header: The header to change
            value: The value to change to

        """
        self.data[header] = value

    def __getitem__(self, key: str) -> Any:
        """Gets the attribute-value for a given key

        Args:
            key: The key to get the data for

        Returns: The value for the given key

        """
        return self.data[key]

    def __str__(self) -> str:
        """Creates a string representation of the tuple
        Returns: A string representation of the tuple

        """
        return self.data.to_string()

    def __eq__(self, other) -> bool:
        """Checks whether two Items are equivalent to each other.

        Args:
            other: The Item to compare to

        Returns: Whether or not the items are equal

        """
        return self.headers == other.headers and self.data.equals(other.data)
