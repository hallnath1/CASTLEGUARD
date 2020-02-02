class Range():

    """Stores the lower and upper values for a cluster on a single axis. """

    def __init__(self, lower=0, upper=0):
        """Initialises the Range with given lower and upper values, or 0s if not provided

        Kwargs:
            lower (TODO): TODO
            upper (TODO): TODO

        Returns: TODO

        """

        self.lower = lower
        self.upper = upper

    def update(self, value):
        """Updates the range if the given value does not fit within the current
        bounds

        Args:
            value (TODO): TODO

        Returns: TODO

        """
        self.lower = min(self.lower, value)
        self.upper = max(self.upper, value)

    def VInfoLoss(self, I):
        """Calculates VinfoLoss of I defined on page 4 of castle paper.

        Args:
            I (Range): Global range of this attribute

        Returns: VInfoLoss of I

        """

        return (self.upper - self.lower) / (I.upper - I.lower)

    def __truediv__(self, I):
        return self.VInfoLoss(I)

    def __str__(self):
        return "Range [{}, {}]".format(self.lower, self.upper)
