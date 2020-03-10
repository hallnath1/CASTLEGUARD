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
        type=int,
        help="The value of k-anonymity to use"
    )

    parser.add_argument(
        "--delta",
        nargs="?",
        type=int,
        help="The maximum number of tuples to store before outputting"
    )

    parser.add_argument(
        "--beta",
        nargs="?",
        type=int,
        help="The maximum number of clusters to allow"
    )

    parser.add_argument(
        "--mu",
        nargs="?",
        type=int,
        help="The number of most recent loss values to use for tau"
    )

    parser.add_argument(
        "--l",
        nargs="?",
        type=int,
        help="The value of l to use"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="The random seed to use for the simulation"
    )

    parser.add_argument(
        "--sample-size",
        nargs="?",
        default=20,
        type=int,
        help="The number of samples to use for the simulation"
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

    parser.add_argument(
        "--disable-dp",
        action="store_false"
    )

    parser.add_argument(
        "--phi",
        nargs="?",
        type=int,
        help="phi value used to perturb tuples for k anonymity"
    )

    parser.add_argument(
        "--big-beta",
        nargs="?",
        type=float,
        help="drop input tuples with probability 1 - beta"
    )

    return parser

def parse_args() -> argparse.Namespace:
    """Parses the arguments specified on the command line
    Returns: The arguments that were specified on the command line and parsed
    by the object returned by build_parser()

    """
    parser = build_parser()
    return parser.parse_args()
