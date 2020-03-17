# CASTLEGUARD

[![Build Status](https://travis-ci.com/hallnath1/CASTLEGUARD.svg?token=CbJDgsGLo7GCV1xLzUAy&branch=master)](https://travis-ci.com/hallnath1/CASTLEGUARD)

CASTLEGUARD (Continuously Anonymizing STreaming data via adaptive cLustEring
with GUARanteed Differential privacy) is an extension to the CASTLE data stream
anonymisation algorithm.

> Data streams are a common tool used by data controllers to outsource data
> processing of time series data to external data processors \cite{qiu_2008}.
> Data protection legislation now enforces that data controllers are responsible
> for providing a guarantee of privacy for user data contained within published
> data streams \cite{eudgpr_2017}. CASTLE \cite{cao_2011} is a well-known method
> of anonymising data streams with a guarantee of $k$-anonymity, however,
> $k$-anonymity has been shown to be a weak privacy guarantee with numerous
> vulnerabilities in practice. As such, users may desire a stronger privacy
> guarantee. We propose \textit{Continuously Anonymising STreaming data via
> adaptive cLustEring with GUARanteed Differential privacy} (CASTLEGUARD), a data
> stream anonymisation algorithm which provides a reliable guarantee of
> $l$-diversity and differential privacy based on additional parameters $l$,
> $\beta$ and $\phi$. We achieve differential privacy for data streams by
> sampling entries from an input data stream $S$ with probability $\beta$ and by
> performing "safe" $k$-anonymisation, meaning that no true quasi-identifiable
> attribute value can be inferred from the extreme values of a generalisation,
> using additive noise taken from a Laplace distribution with $\mu = 0$, $b =
> \frac{R}{\phi}$ where $R$ is the range of an attribute. We show that, with
> "safe" $k$-anonymisation and $\beta$-sampling, CASTLEGUARD satisfies
> differentially private $k$-anonymity \cite{li_2011_dp}. Our experimental
> analysis of CASTLEGUARD demonstrates that it effectively protects the
> individual privacy of users whilst still providing effective utility to data
> processors in the context of machine learning.

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
python3 src/main.py --filename example.csv --sample-size 500 --k 25
```

will use `example.csv` as the input stream, pick a random sample of
`500` elements from the dataset and use a `k` value of `25`. All the options
for the interface can be seen using `python3 src/main.py -h`.

The file `example.csv` can be found in the root directory of the
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

| Parameter |                         Meaning                         |    Default Value   |
|:---------:|:-------------------------------------------------------:|:------------------:|
|     k     |        Minimum number of IDs in an output cluster       |          5         |
|   delta   |         Maximum number of tuples in CASTLEGUARD         |         10         |
|    beta   |            Maximum number of active clusters            |          5         |
|     mu    |        Number of information loss values for tau        |          5         |
|     l     | Minimum number of sensitive values in an output cluster |          1         |
|    phi    |                Scale of tuple pertubation               | 100 (must be >= 1) |
|     dp    |        Whether or not to use differential privacy       |        True        |
|  big_beta |           1 - probability of ignoring a tuple           |          1         |

All of these can be configured either in the `Parameters` object or on the
command line.

## Unit Tests

Unit tests can be run using the `pytest` command.
