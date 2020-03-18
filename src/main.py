import numpy as np
import pandas as pd

import app

from castle import CASTLE, Parameters
from visualisations import display_visualisation

def handler(value: pd.Series):
    print("RECIEVED VALUE: \n{}".format(value))

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
