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
        self.lower = min(self.lower, value) if self.lower is not None else value
        self.upper = max(self.upper, value) if self.upper is not None else value

    def VInfoLoss(self, other: Range) -> float:
        """Calculates VInfoLoss of other defined on Page 4 of the CASTLE paper

        Args:
            other (Range): Global range of this attribute

        Returns: VInfoLoss with respect to other

        """
        ds = self.upper - self.lower
        do = other.upper - other.lower

        # Deal with division by 0
        if ds == 0 and do == 0:
            return 0

        return ds / do

    def within_bounds(self, value: float):
        """Checks whether the value is within the bounds of the range.

        Args:
            value: The value to perform bounds checking for

        Returns: Whether or not the value is in bounds

        """
        return self.lower <= value and value <= self.upper

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
