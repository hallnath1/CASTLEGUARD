# CASTLEGUARD

[![Build Status](https://travis-ci.com/hallnath1/CASTLEGUARD.svg?token=CbJDgsGLo7GCV1xLzUAy&branch=master)](https://travis-ci.com/hallnath1/CASTLEGUARD)

CASTLEGUARD stands for Continuously Anonymizing STreaming data via adaptive
cLustEring with GUARanteed Differential privacy and is an extension to the
existing CASTLE algorithm.

> Most of the existing privacy-preserving techniques, such as k-anonymity
> methods, are designed for static data sets. As such, they cannot be applied
> to streaming data which are continuous, transient, and usually unbounded.
> Moreover, in streaming applications, there is a need to offer strong
> guarantees on the maximum allowed delay between incoming data and the
> corresponding anonymized output. To cope with these requirements, in this
> paper, we present Continuously Anonymizing STreaming data via adaptive
> cLustEring (CASTLE), a cluster-based scheme that anonymizes data streams
> on-the-fly and, at the same time, ensures the freshness of the anonymized
> data by satisfying specified delay constraints. We further show how CASTLE
> can be easily extended to handle l-diversity. Our extensive performance study
> shows that CASTLE is efficient and effective w.r.t. the quality of the output
> data.

### Sample Output

![alt text](https://i.imgur.com/Z1dl5jQ.png "An example of data clustering")

## Usage

### Command Line Simulations

The easiest way to play around with `CASTLEGUARD` is by running `src/main.py`.
This will process a stream of data from a file and display the output tuples.
Input filenames, sample sizes, random seeds and hyper-parameters can all be
configured on the command line.

For example:

```bash
python3 src/main.py --filename random_ldiverse.csv --sample-size 500 --k 25
```

will use `random_ldiverse.csv` as the input stream, pick a random sample of
`500` elements from the dataset and use a `k` value of `25`. All the options
for the interface can be seen using `python3 src/main.py -h`.

The file `random_ldiverse.csv` can be found in the root directory of the
project.

### Makefile Examples

A `Makefile` is also provided to give some simple examples of `CASTLEGUARD`,
along with a visualisation of the outputs. For each command, the simulation
will display the tuples and clusters still within `CASTLEGUARD`, along with the
tuples that have been output and the empty clusters.

### Advanced Examples

`CASTLEGUARD` is designed to be easy to use. It requires inputs in the form of
`pandas.Series` objects, which can be obtained by iterating through a
`pandas.DataFrame` object.

Simply add `from castle import CASTLE, Parameters` to a file in `src/`, define
the headers to use for `k-anonymity`, the header to use for `l-diversity`,
create a `Parameters` object and a callback function that takes a
`pandas.Series` object and construct your `CASTLE` object. From here, you can
simply call `CASTLE.insert(element)` to insert data. `CASTLE` will
automatically call your callback function with any generalised tuples it
produces.

#### Available Parameters

The available parameters are as follows:

| Parameter |                         Meaning                         | Sample Value |
|:---------:|:-------------------------------------------------------:|:------------:|
|     k     |        Minimum number of IDs in an output cluster       |       5      |
|   delta   |         Maximum number of tuples in CASTLEGUARD         |      10      |
|    beta   |            Maximum number of active clusters            |       5      |
|     mu    |        Number of information loss values for tau        |       5      |
|     l     | Minimum number of sensitive values in an output cluster |       1      |
|    phi    |                Scale of tuple pertubation               |      100     |
|     dp    |        Whether or not to use differential privacy       |     True     |
|  big_beta |           1 - probability of ignoring a tuple           |       1      |

All of these can be configured either in the `Parameters` object or on the
command line.

## Unit Tests

Unit tests can be run using the `pytest` command.
