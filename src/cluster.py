"""Store tuples that are considered by CASTLE to be together.

Allow for operations on clusters of tuples such as `insert`, `remove` and
`__contains__`. Allow for queries such as the information loss of a cluster,
and the information loss of inserting tuples or merging clusters. Allow for
generalisation of tuples with respect to the cluster itself.
"""

import copy

from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from item import Item
from range import Range

class Cluster():

    """Stores tuples that are considered by the algorithm to be together. """

    def __init__(self, headers: List[str]):
        """Initialises the cluster. """
        self.contents: List[Item] = []
        self.ranges: Dict[str, Range] = {}

        for header in headers:
            self.ranges[header] = Range()

        self.diversity: Set[Any] = set()

        self.sample_values: Dict[str, float] = {}

    def insert(self, element: Item):
        """Inserts a tuple into the cluster

        Args:
            element (Item): The element to insert into the cluster

        """
        self.contents.append(element)

        # Check whether the item is already in a cluster
        if element.parent:
            # If it is, remove it so that we do not reach an invalid state
            element.parent.remove(element)

        # Add sensitive attribute value to the diversity of cluster
        self.diversity.add(element.sensitive_attr)

        # Update the parent of the item to be this cluster
        element.parent = self

        for header, header_range in self.ranges.items():
            header_range.update(element[header])

    def remove(self, element: Item):
        """Removes a tuple from the cluster

        Args:
            element: The element to remove from the cluster

        """
        self.contents.remove(element)

        if not element.sensitive_attr in [e.sensitive_attr for e in self.contents]:
            self.diversity.remove(element.sensitive_attr)

    def generalise(self, item: Item) -> pd.Series:
        """Generalises a tuple based on the ranges for this cluster

        Args:
            item (Item): The tuple to be generalised

        Returns: A generalised version of the tuple based on the ranges for
        this cluster

        """
        gen_tuple = copy.deepcopy(item)

        for header, header_range in self.ranges.items():
            # Pick a random person to use for this attribute
            if not header in self.sample_values:
                self.sample_values[header] = np.random.choice(self.contents)[header]

            gen_tuple.data.loc['min' + header] = header_range.lower
            gen_tuple.data.loc['spc' + header] = self.sample_values[header]
            gen_tuple.data.loc['max' + header] = header_range.upper

            gen_tuple.headers.append('min' + header)
            gen_tuple.headers.append('spc' + header)
            gen_tuple.headers.append('max' + header)

            gen_tuple.headers.remove(header)
            del gen_tuple.data[header]

        del gen_tuple.data['pid']

        return gen_tuple, item

    def tuple_enlargement(self, item: Item, global_ranges: Dict[str, Range]) -> float:
        """Calculates the enlargement value for adding <item> into this cluster

        Args:
            item: The tuple to calculate enlargement based on
            global_ranges: The globally known ranges for each attribute

        Returns: The information loss if we added item into this cluster

        """
        given = self.information_loss_given_t(item, global_ranges)
        current = self.information_loss(global_ranges)

        return (given - current) / len(self.ranges)

    def cluster_enlargement(self, cluster, global_ranges: Dict[str, Range]) -> float:
        """Calculates the enlargement value for merging <cluster> into this cluster

        Args:
            cluster: The cluster to calculate information loss for
            global_ranges: The globally known ranges for each attribute

        Returns: The information loss upon merging cluster with this cluster

        """
        given = self.information_loss_given_c(cluster, global_ranges)
        current = self.information_loss(global_ranges)

        return (given - current) / len(self.ranges)

    def information_loss_given_t(self, item: Item, global_ranges: Dict[str, Range]) -> float:
        """Calculates the information loss upon adding <item> into this cluster

        Args:
            item: The tuple to calculate information loss based on
            global_ranges: The globally known ranges for each attribute

        Returns: The information loss given that we insert item into this cluster

        """
        loss = 0

        # For each range, check if <item> would extend it
        for header, header_range in self.ranges.items():
            global_range = global_ranges[header]
            updated = Range(
                lower=min(header_range.lower, item.data[header]),
                upper=max(header_range.upper, item.data[header])
            )
            loss += updated / global_range

        return loss

    def information_loss_given_c(self, cluster, global_ranges: Dict[str, Range]) -> float:
        """Calculates the information loss upon merging <cluster> into this cluster

        Args:
            cluster: The cluster to calculate information loss based on
            global_ranges: The globally known ranges for each attribute

        Returns: The information loss given that we merge cluster with this cluster

        """
        loss = 0

        # For each range, check if <item> would extend it
        for header, header_range in self.ranges.items():
            global_range = global_ranges[header]
            updated = Range(
                lower=min(header_range.lower, cluster.ranges[header].lower),
                upper=max(header_range.upper, cluster.ranges[header].upper)
            )
            loss += updated / global_range

        return loss

    def information_loss(self, global_ranges: Dict[str, Range]) -> float:
        """Calculates the information loss of this cluster

        Args:
            global_ranges: The globally known ranges for each attribute

        Returns: The current information loss of the cluster

        """
        loss = 0

        for header, header_range in self.ranges.items():
            global_range = global_ranges[header]
            loss += header_range / global_range

        return loss

    def distance(self, other: Item) -> float:
        """Calculates the distance from this tuple to another

        Args:
            other: The tuple to calculate the distance to

        Returns: The distance to the other tuple

        """
        total_distance = 0

        for header, header_range in self.ranges.items():
            total_distance += abs(other[header] - (header_range.difference()))

        return total_distance

    def within_bounds(self, item: Item) -> bool:
        """Checks whether a tuple is within all the ranges of the this
        cluster, eg. would cause no information loss on being entered.

        Args:
            item: The tuple to perform bounds checking on

        Returns: Whether or not the tuple is within the bounds of the cluster

        """
        for header, header_range in self.ranges.items():
            if not header_range.within_bounds(item[header]):
                return False

        return True

    def __len__(self) -> int:
        """Calculates the length of this cluster
        Returns: The number of tuples currently contained in the cluster

        """
        return len({item['pid'] for item in self.contents})

    def __contains__(self, item: Item) -> bool:
        """Checks whether this cluster contains item

        Args:
            item: The tuple to find

        Returns: Whether or not this cluster contains item

        """
        return item in self.contents

    def __str__(self) -> str:
        """Creates a string representation of the cluster
        Returns: A string representation of the current cluster

        """
        return "Tuples: {}, Ranges: {}".format(
            str(self.contents),
            str(self.ranges)
        )
