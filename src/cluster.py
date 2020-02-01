class Cluster():

    """Stores tuples that are considered by the algorithm to be together. """

    def __init__(self):
        """Initialises the cluster """
        self.contents = []
        self.ranges = []

    def insert(self, element):
        """Inserts a tuple into the cluster

        Args:
            element (TODO): TODO

        Returns: TODO

        """
        self.contents.append(element)

    def enlargement(self, t):
        """Calculates the enlargement value for adding <t> into this cluster

        Args:
            t (TODO): TODO

        Returns: TODO

        """
        pass

    def information_loss(self, t):
        """Calculates the information loss upon adding <t> into this cluster

        Args:
            t (TODO): TODO

        Returns: TODO

        """
        pass

    def __len__(self):
        return len(self.contents)

    def __contains__(self, item):
        return item in self.contents
