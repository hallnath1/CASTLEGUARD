import argparse

def build_parser():
    """Builds the command line parser
    Returns: TODO

    """
    parser = argparse.ArgumentParser(
        description="CLI interface for the CASTLE algorithm."
    )

    parser.add_argument(
        "--k",
        nargs="?",
        default=5,
        type=int,
        help="The value of k-anonymity to use"
    )

    parser.add_argument(
        "--delta",
        nargs="?",
        default=10,
        type=int,
        help="The maximum number of tuples to store before outputting"
    )

    parser.add_argument(
        "--beta",
        nargs="?",
        default=5,
        type=int,
        help="The maximum number of clusters to allow"
    )

    parser.add_argument(
        "-f", "--filename",
        nargs="?",
        default="data.csv",
        type=str,
        help="The filepath to read data from"
    )

    parser.add_argument(
        "--display",
        action="store_true"
    )

    return parser

def parse_args():
    """Parses the arguments specified on the command line
    Returns: TODO

    """
    parser = build_parser()
    return parser.parse_args()
