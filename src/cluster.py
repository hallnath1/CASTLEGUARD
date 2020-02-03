import copy
import pandas as pd

from range import Range

class Cluster():

    """Stores tuples that are considered by the algorithm to be together. """

    def __init__(self, headers):
        """Initialises the cluster """
        self.contents = []
        self.ranges = {}

        for header in headers:
            self.ranges[header] = Range()

    def insert(self, element):
        """Inserts a tuple into the cluster

        Args:
            element (TODO): TODO

        Returns: TODO

        """
        self.contents.append(element)
        element.parent = self

        # TODO: update ranges using element headers

        for k, v in self.ranges.items():
            v.update(element[k])

    def enlargement(self, t, global_ranges):
        """Calculates the enlargement value for adding <t> into this cluster

        Args:
            t (TODO): TODO

        Returns: TODO

        """

        info_loss = self.information_loss_given(t, global_ranges) - self.information_loss(global_ranges)

        return info_loss / len(self.ranges)

    def information_loss_given(self, t, global_ranges):
        """Calculates the information loss upon adding <t> into this cluster

        Args:
            t (TODO): TODO

        Returns: TODO

        """
        loss = 0

        # For each range, check if <t> would extend it
        for k, r in self.ranges.items():
            global_range = global_ranges[k]
            updated = Range(
                lower=min(r.lower, t[k]),
                upper=max(r.upper, t[k])
            )
            loss += updated / global_range

        return loss

    def information_loss(self, global_ranges):
        """Calculates the information loss of this cluster

        Returns: information loss of cluster

        """
        loss = 0
        for k, r in self.ranges.items():
            global_range = global_ranges[k]
            loss += r / global_range

        return loss

    def __len__(self):
        return len(self.contents)

    def __contains__(self, item):
        return item in self.contents

    def __str__(self):
        return "Tuples: {}, Ranges: {}".format(
            str(self.contents),
            str(self.ranges)
        )
