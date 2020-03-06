import pandas as pd

import util_graphs as util

if __name__ == "__main__":
    frame = pd.read_csv("data.csv")
    headers = list(frame.columns.values)

    # Plot Beta Variation
    util.test_beta("data.csv", [10])

    # Plot Beta and Mu Variation
    util.test_beta_mu("data.csv", [10,20], [10,20])
    # Graph x,y,z ...
