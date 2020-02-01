class Range():

    """Stores the lower and upper values for a cluster on a single axis. """

    def __init__(self, lower, upper):
        """Initialises the Range with given lower and upper values

        Args:
            lower (Number): Lower bound of range
            upper (Number): Upper bound of range

        """

        self.lower = lower
        self.upper = upper

    def extend(self, lower=self.lower, upper=self.upper):
        """Extends the range of the object optionally

        Kwargs:
            lower (TODO): New Lower bound of range
            upper (TODO): New Upper bound of range

        Returns: TODO

        """
        pass

    def VInfoLoss(self, I):
        """Calculates VinfoLoss of I defined on page 4 of castle paper.

        Args:
            I (Range): Global range of this attribute

        Returns: VInfoLoss of I

        """

        return (self.upper - self.lower) / (I.upper - I.lower)
