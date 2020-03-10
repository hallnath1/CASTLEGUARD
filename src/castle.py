import math

import numpy as np
import pandas as pd

from typing import Any, Callable, Deque, Dict, List, Optional
from collections import deque
from dataclasses import dataclass

from cluster import Cluster
from range import Range
from item import Item

@dataclass
class Parameters:
    k: int
    delta: int
    beta: int
    mu: int
    l: int
    phi: int
    dp: bool
    big_beta: float

class CASTLE():

    """An implementation of the CASTLE Algorithm designed by Jianneng Cao,
    Barbara Carminati, Elena Ferrari and Kian-Lee Tan."""

    def __init__(self, callback: Callable[[pd.Series], None], headers: List[str], sensitive_attr: str, params: Parameters):
        """Initialises the CASTLE algorithm with necessary parameters.

        Args:
            callback: The function to call when a tuple is ejected
            headers: The columns that need to be anonymised according to the
            algorithm
            k: The level of anonymity to provide
            delta: The maximum number of active tuples at a time
            beta: The maximum number of active clusters at a time
        """
        self.callback: Callable[[pd.Series], None] = callback

        self.deque: Deque = deque()
        self.headers: List[str] = headers
        self.sensitive_attr: str = sensitive_attr

        # Required number of tuples for a cluster to be complete
        self.k: int = params.k
        # Maximum number of active tuples
        self.delta: int = params.delta
        # Maximum number of clusters that can be active
        self.beta: int = params.beta
        # Maximum amount of information loss, by default set to infinity as we
        # have no clusters
        self.tau: float = math.inf
        # Number of values to use in the rolling average
        self.mu: int = params.mu
        # Required number of distinct sensitive attributes for a cluster to be complete
        self.l: int = params.l

        # Whether we want to enable differential privacy
        self.dp: bool = params.dp

        if self.dp:
            # The 'scale' of tuple fudging
            self.phi: int = params.phi
            # The percentage chance of ignoring a tuple
            self.big_beta: float = params.big_beta

        # Set of non-ks anonymised clusters
        self.big_gamma: List[Cluster] = []
        # Set of ks anonymised clusters
        self.big_omega: List[Cluster] = []

        # Global ranges for the data stream S
        self.global_ranges: Dict[str, Range] = {}
        # Range for the sensitive attribute
        self.sensitive_range = Range()
        # Initialise them as empty ranges
        for header in self.headers:
            self.global_ranges[header] = Range()

        # Deque of all tuple objects and parent cluster pairs
        self.global_tuples: Deque = deque()

        # Rolling values of information loss for tau
        self.recent_losses = []

    def update_global_ranges(self, data: Item):
        """Updates the globally known ranges for each column based on the value
        that this Item contains

        Args:
            data: The new element of data that has just been inserted into the
            algorithm

        """
        # Update the sensitive attribute range
        self.sensitive_range.update(data.data[self.sensitive_attr])

        for header in self.headers:
            self.global_ranges[header].update(data.data[header])

    def insert(self, data: pd.Series):
        """Inserts a new piece of data into the algorithm and its state

        Args:
            data: The element of data to insert into the algorithm

        """
        # Probabilty 1 - beta tuple is never inserted
        if np.random.rand() > self.big_beta and self.dp:
            return

        # Update the global range values
        item = Item(data=data, headers=self.headers, sensitive_attr=self.sensitive_attr)
        self.update_global_ranges(item)

        # If we are using differential privacy, fudge our tuples
        if self.dp:
            # Perturb tuple to satsify k anonymity
            self.fudge_tuple(item)

        cluster = self.best_selection(item)

        if not cluster:
            # Create a new cluster
            cluster = Cluster(self.headers)
            self.big_gamma.append(cluster)

        cluster.insert(item)
        self.global_tuples.append(item)

        # If we now have too many tuples, try and output one
        if len(self.global_tuples) > self.delta:
            # Get the next tuple to be output
            t_prime = self.global_tuples[0]
            # print("Attempting to output: \n{}".format(t_prime))
            self.delay_constraint(t_prime)

        self.update_tau()

    def fudge_tuple(self, t: Item):
        """ Fudges a tuple based on laplace distribution

        Args:
            tuple: The tuple to be perturbed

        """

        for header in self.headers:
            if self.global_ranges[header].lower is not None and self.global_ranges[header].upper is not None:
                scale = max(self.global_ranges[header].difference(), 1) / self.phi
                dist = np.round(np.random.laplace(scale=scale))
                original_value = t.data[header]
                t.update_attribute(header, original_value + dist)

        if self.sensitive_range.lower is not None and self.sensitive_range.upper is not None:
            scale = max(self.sensitive_range.difference(), 1) / self.phi
            dist = np.round(np.random.laplace(scale=scale))
            original_value = t.data[self.sensitive_attr]
            t.update_attribute(self.sensitive_attr, original_value + dist)

    def output_cluster(self, c: Cluster):
        """Outputs a cluster according to the algorithm

        Args:
            c: The cluster to output with generalisations

        """
        # Get the number of unique PIDs in the cluster
        unique_pids = len(set(t['pid'] for t in c.contents))
        sc = [c] if unique_pids < 2 * self.k and len(c.diversity) < self.l else self.split_l(c)
        for cluster in sc:
            for t in cluster.contents:
                [generalised, original_tuple] = cluster.generalise(t)
                self.suppress_tuple(original_tuple)
                self.callback(generalised)

            # Calculate the information loss of the cluster
            info_loss = cluster.information_loss(self.global_ranges)
            self.recent_losses.append(info_loss)

            # If there are too many elements in here, remove one
            if len(self.recent_losses) > self.mu:
                self.recent_losses.pop(0)

            self.update_tau()

            # TODO: Update self.tau according to infoLoss(cluster) #
            # TODO: Decide whether to delete cluster or move to self.big_omega #
            # TODO: This should probably happen #
            # self.big_gamma.remove(cluster)

        # for t in cluster.contents:
        #     print("FOR T")
        #     self.callback(t)

    def update_tau(self):
        self.tau = math.inf

        # If we have elements in recent_losses, take an average
        if self.recent_losses:
            self.tau = sum(self.recent_losses) / len(self.recent_losses)
        elif self.big_gamma:
            # Get 5 elements if we have them, otherwise just get all of them
            sample_size = min(len(self.big_gamma), 5)
            chosen = np.random.choice(self.big_gamma, size=sample_size)

            # Sum the information loss for each chosen cluster
            total_loss = sum(c.information_loss(self.global_ranges) for c in chosen)
            self.tau = total_loss / sample_size

    def suppress_tuple(self, t: Item):
        """Suppresses a tuple from being output and deletes it from the CASTLE
        state. Removes it from the global tuple queue and also the cluster it
        is being contained in

        Args:
            t: The tuple to suppress

        """
        # Remove the tuple from the global queue
        self.global_tuples.remove(t)
        # Remove the tuple from its cluster
        containing_cluster = t.parent
        containing_cluster.remove(t)

    def best_selection(self, t: Item) -> Optional[Cluster]:
        """Finds the best matching cluster for <element>

        Args:
            t: The tuple to find the best cluster for

        Returns: Either a cluster for t to be inserted into, or None if a new
        cluster should be created

        """
        e = set()

        for cluster in self.big_gamma:
            e.add(cluster.tuple_enlargement(t, self.global_ranges))

        # If e is empty, we should return None so a new cluster gets made
        if not e:
            return None

        minima = min(e)
        setCmin = [cluster for cluster in self.big_gamma if
                   cluster.tuple_enlargement(t, self.global_ranges) == minima]

        setCok = set()

        for cluster in setCmin:
            ilcj = cluster.information_loss_given_t(t, self.global_ranges)
            if ilcj <= self.tau:
                setCok.add(cluster)

        if not setCok:
            if self.beta <= len(self.big_gamma):
                return np.random.choice(tuple(setCmin))
            else:
                return None
        else:
            return np.random.choice(tuple(setCok))

        return None

    def delay_constraint(self, t: Item):
        """Decides whether to suppress <t> or not

        Args:
            t: The tuple to make decisions based on

        """
        if self.k <= len(t.parent) and self.l < len(t.parent.diversity):
            self.output_cluster(t.parent)
            return

        # Get all the clusters that t could be within
        KCset = [c for c in self.big_omega if c.within_bounds(t)]

        if KCset:
            # Pick a random cluster from the set and generalise, then output
            random_cluster = np.random.choice(KCset)
            generalised, original = random_cluster.generalise(t)
            self.suppress_tuple(original)
            return self.callback(generalised)

        m = 0

        for cluster in self.big_gamma:
            if len(t.parent) < len(cluster):
                m += 1

        if m > len(self.big_gamma) / 2:
            return self.suppress_tuple(t)

        total_tuples = sum([len(cluster) for cluster in self.big_gamma])
        diversity_values = set()

        for cluster in self.big_gamma:
            diversity_values.update(cluster.diversity)

        if total_tuples < self.k or len(diversity_values) < self.l:
            return self.suppress_tuple(t)

        mc = self.merge_clusters(t.parent)

        self.output_cluster(mc)

    # TODO: Check this function is correct #
    def split(self, c: Cluster) -> List[Cluster]:
        """Splits a cluster <c>

        Args:
            c: The cluster that needs to be split into smaller clusters

        Returns: List of new clusters with tuples inside them

        """
        sc = []

        # Group everyone by pid
        buckets: Dict[int, List[Item]] = {}

        # Insert all the tuples into the relevant buckets
        for t in c.contents:
            if t.data.pid not in buckets:
                buckets[t.data.pid] = []

            buckets[t.data.pid].append(t)

        # While k <= number of buckets
        while self.k <= len(buckets):
            # Pick a random tuple from a random bucket
            pid = np.random.choice(list(buckets.keys()))
            bucket = buckets[pid]
            t = bucket.pop(np.random.randint(0, len(bucket)))

            # Create a new subcluster over t
            cnew = Cluster(self.headers)
            cnew.insert(t)

            # Check whether the bucket is empty
            if not buckets[pid]:
                del buckets[pid]

            heap = []

            for key, value in buckets.items():
                if key == pid:
                    continue

                # Pick a random tuple in the bucket
                random_tuple = np.random.choice(value)

                # Insert the tuple to the heap
                heap.append(random_tuple)

            # Sort the heap by distance to our original tuple
            distance_func = lambda t2: t.tuple_distance(t2)
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
            ti = np.random.choice(bi)

            # Find the nearest cluster in sc
            nearest = min(sc, key=lambda c: c.tuple_enlargement(ti, self.global_ranges))

            for t in bi:
                nearest.insert(t)

        return sc

    # TODO: Check this function is correct #
    def split_l(self, C: Cluster) -> List[Cluster]:
        """Splits a cluster <c> ensuring l-diversity

        Args:
            C: The cluster that needs to be split into smaller clusters

        Returns: List of new clusters with tuples inside them

        """
        sc = []

        # Group every tuple by the sensitive attribute
        buckets = self.generate_buckets(C)

        # if number of buckets is less than l cannot split
        if len(buckets) < self.l:
            return {C}

        # While length of buckets greater than l and more than k tuples
        while len(buckets) >= self.l and sum([len(b) for b in buckets.values()]) >= self.k:
            # Pick a random tuple from a random bucket
            pid = np.random.choice(list(buckets.keys()))
            bucket = buckets[pid]
            t = bucket.pop(np.random.randint(0, len(bucket)))

            # Create a new subcluster over t
            cnew = Cluster(self.headers)
            cnew.insert(t)

            # Delete t from b
            del buckets[pid]

            for bucket in buckets.values():
                # Sort the bucket by the enlargement value of that cluster
                sorted_bucket = sorted(bucket, key=lambda t: C.tuple_enlargement(t, self.global_ranges))

                # Count the number of tuples we have
                total_tuples = sum([len(b) for b in buckets.values()])
                # Calculate the number of tuples we should take
                chosen_count = int(self.k * (len(sorted_bucket) / total_tuples))
                # Get the subset of tuples
                subset = sorted_bucket[:chosen_count]

                # Insert the top Tj tuples in a new cluster
                for t in subset:
                    cnew.insert(t)
                    bucket.remove(t)

                # if bucket is empty delete the bucket
                if not bucket:
                    del bucket

            sc.append(cnew)

        # For all remaining tuples in this cluster add them to the nearest cluster
        for bucket in buckets.values():
            for t in bucket:
                cluster = min(sc, key=lambda c: c.distance(t))
                cluster.insert(t)

            del bucket

        # This is in the pseudo code
        for c in sc:
            for t in c.contents:
                G = [t_h for t_h in C.contents if t_h['pid'] == t['pid']]
                for g in G:
                    c.insert(t)

            self.big_gamma.append(c)

        return sc

    def generate_buckets(self, c: Cluster) -> Dict[Any, List[Item]]:
        """Groups all tuples in the cluster by their sensitive attribute

        Args:
            c: The cluster to generate the buckets for

        Returns: A dictionary of attribute values to lists of items with those
        values

        """
        buckets: Dict[Any, List[Item]] = {}

        # Insert all the tuples into the relevant buckets
        for t in c.contents:
            # Get the value for the sensitive attribute for this tuple
            sensitive_value = t[self.sensitive_attr]

            # If it isn't in our dictionary, make an empty list for it
            if sensitive_value not in buckets:
                buckets[sensitive_value] = []

            # Insert the tuple into the cluster
            buckets[sensitive_value].append(t)

        return buckets

    def merge_clusters(self, c: Cluster) -> Cluster:
        """Merges a cluster with other clusters in big_gamma until the size of
        the resulting cluster is larger than k

        Args:
            c: The cluster that needs to be merged

        Returns: A cluster with a size larger than or equal to k

        """
        gamma_c = [cluster for cluster in self.big_gamma if cluster != c]

        while len(c) < self.k:
            # Get the cluster with the lowest enlargement value
            lowest_enlargement_cluster = min(gamma_c, key=lambda cl: c.cluster_enlargement(cl, self.global_ranges))
            items = [t for t in lowest_enlargement_cluster.contents]

            for t in items:
                c.insert(t)

            self.big_gamma.remove(lowest_enlargement_cluster)
            gamma_c.remove(lowest_enlargement_cluster)

        return c
