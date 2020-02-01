import time

import pandas as pd

from castle import CASTLE

def handler(value):
    print("RECIEVED VALUE: {}".format(value))

def main():
    frame = pd.read_csv("data.csv")

    headers = list(frame.columns.values)
    stream = CASTLE(handler, headers, 5, 10, 5)

    for (_, row) in frame.iterrows():
        stream.insert(row)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
