import random

from typing import Any, Callable, Deque, Tuple
from collections import deque

from cluster import Cluster

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

        # Set of non-ks anonymised clusters
        self.big_gamma: List[Cluster] = []
        # Set of ks anonymised clusters
        self.big_omega: List[Cluster] = []

    def insert(self, data: Any):
        cluster = self.best_selection(data)

        if not cluster:
            # Create a new cluster
            cluster = Cluster()
            self.big_gamma.append(cluster)

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

    def best_selection(self, element):
        """Finds the best matching cluster for <element>

        Args:
            element (TODO): TODO

        Returns: TODO

        """
        e = {}

        for cluster in self.big_gamma:
            e.insert(cluster.enlargement(t))

        minima = min(e)
        setCmin = [cluster for cluster in self.big_gamma if cluster.enlargement(t) == minima]

        setCok = set()

        for cluster in setCmin:
            ilcj = cluster.information_loss(t)
            if ilcj <= self.tau:
                setCok.insert(cluster)

        if not setCok:
            if self.beta <= len(self.big_gamma):
                # TODO: Return any cluster in setCmin with minimal size #
                pass
            else:
                return None
        else:
            # TODO: Return any cluster in setCok with minimal size #
            pass

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
            cnew = Cluster()
            cnew.insert(t)

            # Check whether the bucket is empty
            if not buckets[pid]:
                del buckets[pid]
