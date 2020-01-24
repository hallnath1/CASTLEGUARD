from typing import Any, Callable, Deque, Tuple
from collections import deque

class CASTLE():

    """Docstring for CASTLE. """

    def __init__(self, callback: Callable[[Any], None], headers: Tuple[str]):
        """TODO: to be defined.

        Args:
            callback (TODO): TODO
        """

        self.callback: Callable[[Any], None] = callback

        self.deque: Deque = deque()
        self.delta: int = 5
        self.headers: Tuple[str] = headers
        print("self.headers: {}".format(self.headers))

    def insert(self, data: Any):
        print("INSERTING {}".format(data))
        self.deque.append(data)

        # Check whether we are over the bounds size
        if self.delta <= len(self.deque):
            self.output()

    def output(self):
        element = self.deque.popleft()
        print("OUTPUTTING: {}".format(element))
        self.callback(element)
