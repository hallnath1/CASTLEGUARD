import sys
import traceback

import numpy as np
import pandas as pd

import app

from castle import CASTLE, Parameters
from tqdm import tqdm

def handler(value: pd.Series):
    pass

def generate_parameters(args):
    """Generates some random parameters for the program to use

    Args:
        args: The arguments supplied to the program

    Returns: Randomly generate parameters

    """
    p = Parameters()

    p.k = np.random.randint(1, 100),
    p.delta = np.random.randint(1, 100),
    p.beta = np.random.randint(1, 100),
    p.mu = np.random.randint(1, 100),
    p.l = np.random.randint(1, 10)

    return p

def main():
    # Parse the regular arguments for the program
    args = app.parse_args()
    print("args: {}".format(args))

    # Set a random seed
    seed = np.random.randint(1e6)
    np.random.seed(seed)
    print("seed: {}".format(seed))

    # Generate some parameters
    params = generate_parameters(args)
    print("params: {}".format(params))

    # Read the file contents
    frame = pd.read_csv(args.filename).sample(args.sample_size)

    headers = list(frame.columns.values)[1:-1]
    print("headers: {}".format(headers))
    sensitive_attr = headers[-1]
    print("sensitive_attr: {}".format(sensitive_attr))
    stream = CASTLE(handler, headers, sensitive_attr, params)

    try:
        for (_, row) in tqdm(frame.iterrows()):
            stream.insert(row)
    except Exception as e:
        traceback.print_exc()
        print(seed, params, args.filename, headers, sensitive_attr)
        sys.exit(1)

if __name__ == "__main__":
    main()
