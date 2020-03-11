import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import app

from castle import CASTLE, Parameters
from range import Range

def handler(value: pd.Series):
    print("RECIEVED VALUE: \n{}".format(value))

def create_rectangle(rx: Range, ry: Range) -> patches.Rectangle:
    width = rx.upper - rx.lower
    height = ry.upper - ry.lower
    return patches.Rectangle((rx.lower, ry.lower), width, height, fill=False)

def display_visualisation(stream: CASTLE):

    cm = plt.cm.get_cmap('viridis')

    ax = plt.subplot(1, 2, 1)
    plt.title("CASTLE Ω Clusters and Output Tuples (Φ = {}, β = {})".format(stream.phi, stream.big_beta))
    plt.xlabel("Pickup Location ID")
    plt.ylabel("Trip Distance")

    for cluster in stream.big_gamma:
        plid_range = cluster.ranges["PickupLocationID"]
        distance_range = cluster.ranges["TripDistance"]
        ax.add_patch(create_rectangle(plid_range, distance_range))

    pickup_id = [t["PickupLocationID"] for t in stream.global_tuples]
    distance = [t["TripDistance"] for t in stream.global_tuples]
    fare_amount = [t.sensitive_attr for t in stream.global_tuples]
    s = plt.scatter(pickup_id, distance, c=fare_amount, cmap=cm)
    plt.colorbar(s)

    ax = plt.subplot(1, 2, 2)
    plt.title("CASTLE Ω Clusters and Output Tuples (Φ = {}, β = {})".format(stream.phi, stream.big_beta))
    plt.xlabel("Pickup Location ID")
    plt.ylabel("Trip Distance")


    for cluster in stream.big_omega:
        plid_range = cluster.ranges["PickupLocationID"]
        distance_range = cluster.ranges["TripDistance"]

        ax.add_patch(create_rectangle(plid_range, distance_range))

    pickup_id = [t["PickupLocationID"] for t in stream.history]
    distance = [t["TripDistance"] for t in stream.history]
    fare_amount = [t.sensitive_attr for t in stream.history]
    s = plt.scatter(pickup_id, distance, c=fare_amount, cmap=cm)
    plt.colorbar(s)
    plt.show()

def main():
    args = app.parse_args()
    print("args: {}".format(args))

    seed = args.seed if args.seed else np.random.randint(1e6)
    np.random.seed(seed)
    print("USING RANDOM SEED: {}".format(seed))

    frame = pd.read_csv(args.filename).sample(args.sample_size)

    headers = ["PickupLocationID", "TripDistance"]
    params = Parameters(args)
    sensitive_attr = "FareAmount"

    stream = CASTLE(handler, headers, sensitive_attr, params)

    for (_, row) in frame.iterrows():
        stream.insert(row)

    if args.display:
        display_visualisation(stream)

if __name__ == "__main__":
    main()
