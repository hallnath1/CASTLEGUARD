import time

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from castle import CASTLE

def handler(value):
    print("RECIEVED VALUE: {}".format(value))

def create_rectangle(rx, ry):
    width = rx.upper - rx.lower
    height = ry.upper - ry.lower
    return patches.Rectangle((rx.lower, ry.lower), width, height, fill=False)

def main():
    frame = pd.read_csv("data.csv").sample(20)

    headers = list(frame.columns.values)
    stream = CASTLE(handler, headers, 5, 10, 5)

    for (_, row) in frame.iterrows():
        stream.insert(row)
        # time.sleep(0.5)

    fig, ax = plt.subplots(1)

    for cluster in stream.big_gamma:
        print([str(r) for r in cluster.ranges.values()])
        plid_range = cluster.ranges["PickupLocationID"]
        distance_range = cluster.ranges["TripDistance"]

        ax.add_patch(create_rectangle(plid_range, distance_range))

        for t in cluster.contents:
            pickup_id = t["PickupLocationID"]
            distance = t["TripDistance"]
            print(pickup_id, distance)
            plt.scatter(pickup_id, distance)

    plt.show()

if __name__ == "__main__":
    main()
