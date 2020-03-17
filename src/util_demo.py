import pandas as pd

import util_graphs as util

if __name__ == "__main__":
    frame = pd.read_csv("data.csv")
    headers = list(frame.columns.values)

    # Plot Beta and Mu Variation
    util.test_beta_mu("example.csv", [50, 100, 150, 200, 250], [50, 100, 150, 200, 250])

    # Graph x,y,z ...
