import argparse

def build_parser() -> argparse.ArgumentParser:
    """Builds the command line parser
    Returns: An argument parser ready to parse arguments for this program

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
        "--mu",
        nargs="?",
        default=5,
        type=int,
        help="The number of most recent loss values to use for tau"
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

def parse_args() -> argparse.Namespace:
    """Parses the arguments specified on the command line
    Returns: The arguments that were specified on the command line and parsed
    by the object returned by build_parser()

    """
    parser = build_parser()
    return parser.parse_args()
