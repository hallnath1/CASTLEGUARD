import math

class Item:

    def __init__(self, data, headers):
        self.data = data
        self.headers = headers

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self):
        return self.data.to_string()

    def __eq__(self, i):
        return self.headers == i.headers and self.data.equals(i.data)

    def tuple_distance(self, t):
        """Calculates the distance between the two tuples

        Args:
            t (TODO): TODO

        Returns: TODO

        """
        s = self.data[self.headers]
        t = t.data[self.headers]
        error = s.sub(t).abs()
        mean_squared_error = error.pow(2).mean(axis=0)
        return math.sqrt(mean_squared_error)
