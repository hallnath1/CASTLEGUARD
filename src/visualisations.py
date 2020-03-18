import matplotlib.pyplot as plt
import matplotlib.patches as patches

from castle import CASTLE
from range import Range

def create_rectangle(rx: Range, ry: Range) -> patches.Rectangle:
    """Creates a rectangle for the ranges given

    Args:
        rx: The range for the x-axis
        ry: The range for the y-axis

    Returns: A patches.Rectangle object defining the rectangle

    """
    width = rx.upper - rx.lower
    height = ry.upper - ry.lower
    return patches.Rectangle((rx.lower, ry.lower), width, height, fill=False)

def display_visualisation(stream: CASTLE):
    """Displays the internal and external state of CASTLE

    Args:
        stream: An instance of CASTLE after finishing insertions

    """
    cm = plt.cm.get_cmap('viridis')
    ax = plt.subplot(1, 2, 1) if stream.history else plt.subplot(1, 1, 1)

    xlabel, ylabel = stream.headers

    plt.title("CASTLE Ω Clusters and Output Tuples (Φ = {}, β = {})".format(stream.phi, stream.big_beta))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for cluster in stream.big_gamma:
        plid_range = cluster.ranges[xlabel]
        distance_range = cluster.ranges[ylabel]
        ax.add_patch(create_rectangle(plid_range, distance_range))

    pickup_id = [t[xlabel] for t in stream.global_tuples]
    distance = [t[ylabel] for t in stream.global_tuples]
    fare_amount = [t.sensitive_attr for t in stream.global_tuples]
    s = plt.scatter(pickup_id, distance, c=fare_amount, cmap=cm)
    plt.colorbar(s)

    if stream.history:
        ax = plt.subplot(1, 2, 2)
        plt.title("CASTLE Ω Clusters and Output Tuples (Φ = {}, β = {})".format(stream.phi, stream.big_beta))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        for cluster in stream.big_omega:
            plid_range = cluster.ranges[xlabel]
            distance_range = cluster.ranges[ylabel]

            ax.add_patch(create_rectangle(plid_range, distance_range))

        pickup_id = [t[xlabel] for t in stream.tuple_history]
        distance = [t[ylabel] for t in stream.tuple_history]
        fare_amount = [t.sensitive_attr for t in stream.tuple_history]
        s = plt.scatter(pickup_id, distance, c=fare_amount, cmap=cm)
        plt.colorbar(s)

    plt.show()

