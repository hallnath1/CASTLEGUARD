import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import app
from castle import CASTLE, Parameters

"""
    Note: Not all tuples are outputted due to algorithm assuming a continuous stream, 
    so some inputs are ignored in the calculation
"""

latency_dict = {}
latency_list = []

def handler(value: pd.Series):

    latency_list.append(time.time()-latency_dict[value["pid"]])

    print("RECIEVED VALUE: \n{}".format(value))
    

def jitter_wrapper(params, frame):
    headers = ["PickupLocationID", "TripDistance"]
    
    stream = CASTLE(handler, headers, params)
      
    for (_, row) in frame.iterrows():
        
        latency_dict[row["pid"]] = time.time()
        stream.insert(row)

    jitter = 0
    for i in range(len(latency_list)-1):
        jitter += abs(latency_list[i] - latency_list[i+1])
    jitter /= len(latency_list)-1

    return jitter

if __name__ == "__main__":
    args = app.parse_args()
    print("args: {}".format(args))

    seed = args.seed if args.seed else np.random.randint(1e6)
    np.random.seed(seed)
    print("USING RANDOM SEED: {}".format(seed))

    frame = pd.read_csv(args.filename).sample(20)

    params = Parameters(args)
    jitter = jitter_wrapper(params, frame)
    print("JITTER: {}s".format(jitter))
    


        

    

    

 
  
 


