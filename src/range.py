class Range():

    """Stores the lower and upper values for a cluster on a single axis. """

    def __init__(self, lower, upper):
        """Initialises the Range with given lower and upper values

        Args:
            lower (TODO): TODO
            upper (TODO): TODO

        """

        self.lower = lower
        self.upper = upper

    def extend(self, lower=self.lower, upper=self.upper):
        """Extends the range of the object optionally

        Kwargs:
            lower (TODO): TODO
            upper (TODO): TODO

        Returns: TODO

        """
        pass
