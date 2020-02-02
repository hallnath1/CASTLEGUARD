import pandas as pd

from range import Range

class Cluster():

    """Stores tuples that are considered by the algorithm to be together. """

    def __init__(self, headers):
        """Initialises the cluster """
        self.contents = pd.DataFrame(columns=headers)
        self.ranges = {}

        for header in headers:
            self.ranges[header] = Range()

    def insert(self, element):
        """Inserts a tuple into the cluster

        Args:
            element (TODO): TODO

        Returns: TODO

        """
        self.contents = self.contents.append(element, ignore_index=True)
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
        # TODO: can be optimised by looping through ranges and calculating infoLoss on ranges that would be updated
        ranges_copy = self.ranges.copy()
        self.insert(t)
        loss = self.information_loss(global_ranges)

        # Get the length of the frame
        length = len(self.contents.index)
        # Remove the last element (the last one we entered)
        self.contents.drop(length - 1)

        self.ranges = ranges_copy

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
