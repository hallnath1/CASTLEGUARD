import pandas as pd
import app
from castle import CASTLE


def handler(value: pd.Series):
    print("RECIEVED VALUE: {}".format(value))

if __name__= "__main__":
    args = app.parse_args()
    
    frame = pd.read_csv(args.filename).sample(20)
    
    headers = list(frame.columns.values)
    
    stream = CASTLE(handler, headers, args.k, args.delta, args.beta)
    
    for(_, row) in frame.iterrows():
        stream.insert(row)


