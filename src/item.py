class Item:

    def __init__(self, data, headers):
        self.data = data
        self.headers = headers

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self):
        return data.to_string()

    # TODO: This is a bad way of calculating distance as it is heavily biased
    # on certain attributes
    #
    # For example, given the original tuple of A = (1, 100) and tuples B = (2,
    # 110) and C = (3, 105), it will find that B is further from A than C is.
    # This might be the desired behaviour however, as if the range of
    # attributes is [1, 3] and [0, 1000], we might want C to be further as it
    # is further along its global range.
    #
    # tuple_distance should use root mean square
    def tuple_distance(self, t):
        """Calculates the distance between the two tuples

        Args:
            t (TODO): TODO

        Returns: TODO

        """
        s = self.data[self.headers]
        t = t.data[self.headers]
        error = s.sub(t).abs()
        mean_squared_error = error.pow(2).mean(axis=1)
        print(mean_squared_error)

        return mean_squared_error.pow(0.5).iloc[0]
