from __future__ import annotations

from typing import Optional

class Range():

    """Stores the lower and upper values for a cluster on a single axis. """

    def __init__(self, lower: Optional[float]=None, upper: Optional[float]=None):
        """Initialises the Range with given lower and upper values, or 0s if not provided

        Kwargs:
            lower (float): The lower bound of the Range
            upper (float): The upper bound of the Range

        """
        self.lower = lower
        self.upper = upper

    def update(self, value: float):
        """Updates the range if the given value does not fit within the current
        bounds

        Args:
            value (float): A new value to bounds check and update the Range
            with if needed

        """
        self.lower = min(self.lower, value) if self.lower else value
        self.upper = max(self.upper, value) if self.upper else value

    def VInfoLoss(self, other: Range) -> float:
        """Calculates VInfoLoss of other defined on Page 4 of the CASTLE paper

        Args:
            other (Range): Global range of this attribute

        Returns: VInfoLoss with respect to other

        """
        return (self.upper - self.lower) / (other.upper - other.lower)

    def __truediv__(self, other: Range):
        """Allows for the shorthand notation r1/r2 instead of r1.VInfoLoss(r2)

        Args:
            other: The other range to use

        Returns: The VInfoLoss of other

        """
        return self.VInfoLoss(other)

    def __str__(self):
        """Creates a string representation of the current Range
        Returns: A string representation of the current Range

        """
        return "Range [{}, {}]".format(self.lower, self.upper)
