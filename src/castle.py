import math, random

from typing import Any, Callable, Deque, Tuple
from collections import deque

from cluster import Cluster
from range import Range

class CASTLE():

    """Docstring for CASTLE. """

    def __init__(self, callback: Callable, headers: Tuple[str], k, delta, beta):
        """TODO: to be defined.

        Args:
            callback (TODO): TODO
        """

        self.callback: Callable = callback

        self.deque: Deque = deque()
        self.headers: Tuple[str] = headers

        # Required number of tuples for a cluster to be complete
        self.k: int = k
        # Maximum number of active tuples
        self.delta: int = delta
        # Maximum number of clusters that can be active
        self.beta: int = beta
        # Maximum amount of information loss, currently set to max
        self.tau: float = -math.inf

        # Set of non-ks anonymised clusters
        self.big_gamma: List[Cluster] = []
        # Set of ks anonymised clusters
        self.big_omega: List[Cluster] = []

        # Global ranges for the data stream S
        self.global_ranges: Dict[str, Range] = {}
        # Initialise them as empty ranges
        for header in self.headers:
            self.global_ranges[header] = Range()

    def update_global_ranges(self, data: Any):
        for header in self.headers:
            self.global_ranges[header].update(data[header])

    def insert(self, data: Any):
        # Update the global range values
        self.update_global_ranges(data)

        cluster = self.best_selection(data)

        if not cluster:
            # Create a new cluster
            cluster = Cluster(self.headers)
            self.big_gamma.append(cluster)

        print("INSERTING {}".format(data))
        cluster.insert(data)

        # Let t' be the tuple with position equal to t.p - delta
        # if t' has not yet been output then
            # delay_constraint(t')

    def output(self):
        element = self.deque.popleft()
        print("OUTPUTTING: {}".format(element))
        self.callback(element)

    def output_cluster(self, c):
        sc = c if len(c) < 2 * self.k else self.split(c)

        for cluster in sc:
            for t in cluster:
                generalised = cluster.generalise(t)
                self.callback(generalised)

            # TODO: Update self.tau according to infoLoss(cluster) #
            # TODO: Decide whether to delete cluster or move to self.big_omega #
            self.big_gamma.remove(cluster)

        for t in cluster:
            self.callback(t)

    def best_selection(self, t):
        """Finds the best matching cluster for <element>

        Args:
            element (TODO): TODO

        Returns: TODO

        """
        e = set()

        for cluster in self.big_gamma:
            e.add(cluster.enlargement(t, self.global_ranges))

        # If e is empty, we should return None so a new cluster gets made
        if not e:
            return None

        minima = min(e)
        setCmin = [cluster for cluster in self.big_gamma if
                   cluster.enlargement(t, self.global_ranges) == minima]

        setCok = set()

        for cluster in setCmin:
            ilcj = cluster.information_loss_given(t, self.global_ranges)
            if ilcj <= self.tau:
                setCok.add(cluster)

        if not setCok:
            if self.beta <= len(self.big_gamma):
                # TODO: Return any cluster in setCmin with minimal size #
                return random.choice(tuple(setCmin))
            else:
                return None
        else:
            # TODO: Return any cluster in setCok with minimal size #
            return random.choice(tuple(setCok))

        return None

    def delay_constraint(self, t):
        """Decides whether to suppress <t> or not

        Args:
            t (TODO): TODO

        Returns: TODO

        """
        # Find the cluster that c belongs to
        for cluster in self.big_gamma:
            if t in cluster:
                c = cluster

        if k <= len(c):
            self.output_cluster(c)
            return

        # Get all the clusters that contain t
        # TODO: This most likely needs to be 'contains', eg within bounds #
        KCset = [cluster for cluster in self.big_omega if t in cluster]

        if KCSet:
            generalised = cluster.generalise(t)
            self.callback(generalised)
            return

        m = 0

        for cluster in self.big_gamma:
            if len(c) < len(cluster):
                m += 1

        if m > len(self.big_gamma) / 2:
            # TODO: Suppress t somehow #
            return

        total_cluster_size = [len(cluster) for cluster in self.big_gamma].sum()

        if total_cluster_size < self.k:
            # TODO: Suppress t somehow #
            return

        diff = [cluster for cluster in self.big_gamma if cluster != c]
        mc = self.merge_clusters(c, diff)

        self.output_cluster(mc)

    # TODO: Check this function is correct #
    def split(self, c):
        """Splits a cluster <c>

        Args:
            c (TODO): TODO

        Returns: TODO

        """
        sc = []

        # Group everyone by pid
        buckets = {}

        # Insert all the tuples into the relevant buckets
        for t in c:
            if t.pid not in buckets:
                buckets[t.pid] = []

            buckets[t.pid].insert(t)

        # While k <= number of buckets
        while k <= len(buckets):
            # Pick a random tuple from a random bucket
            pid = random.choice(list(buckets.keys()))
            bucket = buckets[pid]
            t = bucket.pop(random.randint(0, len(bucket) - 1))

            # Create a new subcluster over t
            cnew = Cluster(self.headers)
            cnew.insert(t)

            # Check whether the bucket is empty
            if not buckets[pid]:
                del buckets[pid]

            heap = []

            for key, value in buckets:
                if key == pid:
                    continue

                # Pick a random tuple in the bucket
                random_tuple = random.choice(value)

                # Insert the tuple to the heap
                heap.append(random_tuple)

            # Sort the heap by distance to our original tuple
            distance_func = lambda t2: self.tuple_distance(t, t2)
            heap.sort(key=distance_func)

            for node in heap:
                cnew.insert(node)

                # Scour the node from the Earth
                containing = [key for key in buckets.keys() if node in buckets[key]]

                for key in containing:
                    buckets[key].remove(node)

                    if not buckets[key]:
                        del buckets[key]


            sc.append(cnew)

        for bi in buckets.values():
            ti = random.choice(bi)

            # Find the nearest cluster in sc
            nearest = min(sc, key=lambda c: c.enlargement(ti, self.global_ranges))

            for t in bi:
                nearest.insert(t)

        return sc

    # TODO: This is a bad way of calculating distance as it is heavily biased
    # on certain attributes
    #
    # For example, given the original tuple of A = (1, 100) and tuples B = (2,
    # 110) and C = (3, 105), it will find that B is further from A than C is.
    # This might be the desired behaviour however, as if the range of
    # attributes is [1, 3] and [0, 1000], we might want C to be further as it
    # is further along its global range.
    def tuple_distance(self, t1, t2):
        """Calculates the distance between the two tuples

        Args:
            t1 (TODO): TODO
            t2 (TODO): TODO

        Returns: TODO

        """
        distance = 0

        for header in self.headers:
            distance += abs(t1[header] - t2[header])

        return distance
