from __future__ import annotations

import copy
import pandas as pd

from typing import Dict, List

from item import Item
from range import Range

class Cluster():

    """Stores tuples that are considered by the algorithm to be together. """

    def __init__(self, headers: List[str]):
        """Initialises the cluster """
        self.contents: List[Item] = []
        self.ranges: Dict[str, Range] = {}

        for header in headers:
            self.ranges[header] = Range()

    def insert(self, element: Item):
        """Inserts a tuple into the cluster

        Args:
            element (Item): The element to insert into the cluster

        """
        self.contents.append(element)
        element.parent = self

        for k, v in self.ranges.items():
            v.update(element[k])

    def generalise(self, t: Item) -> pd.Series:
        """Generalises a tuple based on the ranges for this cluster

        Args:
            t (Item): The tuple to be generalised

        Returns: A generalised version of the tuple based on the ranges for
        this cluster

        """
        gen_tuple = t.copy()
        for h, v in self.ranges.items():
            gen_tuple.data.loc['min' + h] = v.lower
            gen_tuple.data.loc['max' + h] = v.upper
            gen_tuple.headers.append('min' + h)
            gen_tuple.headers.append('max' + h)
            gen_tuple.headers.remove(h)
            del gen_tuple.data[h]

        return t, gen_tuple

    def tuple_enlargement(self, t: Item, global_ranges: Dict[str, Range]) -> float:
        """Calculates the enlargement value for adding <t> into this cluster

        Args:
            t: The tuple to calculate enlargement based on
            global_ranges: The globally known ranges for each attribute

        Returns: The information loss if we added t into this cluster

        """

        info_loss = self.information_loss_given_t(t, global_ranges) - self.information_loss(global_ranges)

        return info_loss / len(self.ranges)

    def cluster_enlargement(self, c: Cluster, global_ranges: Dict[str, Range]) -> float:
        """Calculates the enlargement value for merging <c> into this cluster

        Args:
            c: The cluster to calculate information loss for
            global_ranges: The globally known ranges for each attribute

        Returns: The information loss upon merging c with this cluster

        """

        info_loss = self.information_loss_given_c(c, global_ranges) - self.information_loss(global_ranges)

        return info_loss / len(self.ranges)

    def information_loss_given_t(self, t: Item, global_ranges: Dict[str, Range]) -> float:
        """Calculates the information loss upon adding <t> into this cluster

        Args:
            t: The tuple to calculate information loss based on
            global_ranges: The globally known ranges for each attribute

        Returns: The information loss given that we insert t into this cluster

        """
        loss = 0

        # For each range, check if <t> would extend it
        for k, r in self.ranges.items():
            global_range = global_ranges[k]
            updated = Range(
                lower=min(r.lower, t.data[k]),
                upper=max(r.upper, t.data[k])
            )
            loss += updated / global_range

        return loss

    def information_loss_given_c(self, c, global_ranges: Dict[str, Range]) -> float:
        """Calculates the information loss upon merging <c> into this cluster

        Args:
            c: The cluster to calculate information loss based on
            global_ranges: The globally known ranges for each attribute

        Returns: The information loss given that we merge c with this cluster

        """
        loss = 0

        # For each range, check if <t> would extend it
        for k, r in self.ranges.items():
            global_range = global_ranges[k]
            updated = Range(
                lower=min(r.lower, c.ranges[k].lower),
                upper=max(r.upper, c.ranges[k].upper)
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
        for k, r in self.ranges.items():
            global_range = global_ranges[k]
            loss += r / global_range

        return loss

    def __len__(self) -> int:
        """Calculates the length of this cluster
        Returns: The number of tuples currently contained in the cluster

        """
        return len(self.contents)

    def __contains__(self, t: Item) -> bool:
        """Checks whether this cluster contains t

        Args:
            t: The tuple to find

        Returns: Whether or not this cluster contains t

        """
        return t in self.contents

    def __str__(self) -> str:
        """Creates a string representation of the cluster
        Returns: A string representation of the current cluster

        """
        return "Tuples: {}, Ranges: {}".format(
            str(self.contents),
            str(self.ranges)
        )
