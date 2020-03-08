import pandas as pd

import util_graphs as util

if __name__ == "__main__":

    # Plot Beta Variation
    #util.test_beta("data.csv", [10,20,30,40])

    # Plot Beta and Mu Variation
    util.test_beta_mu("taxis.csv", range(50,200,50), range(50,200,50))
    
    # Graph x,y,z ...
