import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from castle import CASTLE, Parameters

seconds = 0
latency_list = []

def handler(value: pd.Series):
    global seconds
    latency = time.time()-seconds
    latency_list.append(latency)
    print("Cluster Dispactched after {}s".format(latency))
    print("RECIEVED VALUE: \n{}".format(value))
    seconds = time.time()
    

def jitter_wrapper(params, frame):
    global seconds
    frame = pd.read_csv("data.csv")
    headers = list(frame.columns.values)
    
    stream = CASTLE(handler, headers, params)
      
    for (_, row) in frame.iterrows():
        seconds = time.time()  
        stream.insert(row)

    jitter = 0
    for i in range(len(latency_list)-1):
        jitter += abs(latency_list[i] - latency_list[i+1])
    jitter /= len(latency_list)-1

    return jitter

if __name__ == "__main__":
    frame = pd.read_csv("data.csv")
    for k in [5,10,15,20]:
        params = Parameters(k,10,10,10)
        jitter = jitter_wrapper(params, frame)
        print(jitter)


