import time

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import app

from castle import CASTLE

def handler(value):
    print("RECIEVED VALUE: {}".format(value))

def create_rectangle(rx, ry):
    width = rx.upper - rx.lower
    height = ry.upper - ry.lower
    return patches.Rectangle((rx.lower, ry.lower), width, height, fill=False)

def display_visualisation(stream):
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

    frame = pd.read_csv(args.filename).sample(20)

    headers = list(frame.columns.values)
    stream = CASTLE(handler, headers, args.k, args.delta, args.beta)

    for (_, row) in frame.iterrows():
        stream.insert(row)

    if args.display:
        display_visualisation(stream)

if __name__ == "__main__":
    main()
