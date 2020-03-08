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
    fig, ax = plt.subplots(1)

    for cluster in stream.big_gamma:
        plid_range = cluster.ranges["PickupLocationID"]
        distance_range = cluster.ranges["TripDistance"]

        ax.add_patch(create_rectangle(plid_range, distance_range))

        for t in cluster.contents:
            pickup_id = t["PickupLocationID"]
            distance = t["TripDistance"]
            plt.scatter(pickup_id, distance)

    plt.show()

def main():
    args = app.parse_args()
    print("args: {}".format(args))

    seed = args.seed if args.seed else np.random.randint(1e6)
    np.random.seed(seed)
    print("USING RANDOM SEED: {}".format(seed))

    frame = pd.read_csv(args.filename).sample(20)

    headers = ["PickupLocationID", "TripDistance"]
    params = Parameters(args.k, args.delta, args.beta, args.mu, args.l)
    stream = CASTLE(handler, headers, "FareAmount", params)

    for (_, row) in frame.iterrows():
        stream.insert(row)

    if args.display:
        display_visualisation(stream)

if __name__ == "__main__":
    main()
