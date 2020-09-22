# PyTEAP

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI](https://img.shields.io/pypi/v/PyTEAP)](https://pypi.org/project/PyTEAP/)

*PyTEAP* is a Python implementation of [Toolbox for Emotion Analysis using Physiological signals (TEAP)](https://github.com/Gijom/TEAP).

This package intends to reimplement TEAP, originally written in MATLAB, in Python, to enable interoperation with other Python packages.

---

## Installation
To use PyTEAP, you can either clone this repository:
```console
$ git clone https://github.com/cheulyop/PyTEAP.git
$ cd PyTEAP
```
or install it via `pip`:
```console
$ pip install PyTEAP
```

---

## Baseline classification on DEAP
As the primary goal of this package is to provide a toolbox for processing physiological signals for emotion analysis, this package includes a script to perform simple baseline classification on [DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/).

There are two ways you can do baseline classification on DEAP with PyTEAP:

1) Clone this repository and run the script for baseline classification.

```console
$ python baseline.py --root '/path/to/deap_root'
```

Running [baseline.py](https://github.com/cheulyop/PyTEAP/blob/master/baseline.py) will load raw datafiles from the root directory, preprocess features and target labels, perform baseline classification with four simple classifiers: 1) [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html), 2) random voting assuming a uniform distribution between classes, 3) majority voting, 4) class ratio voting, and finally print a table showing performance of each classifier, measured with [accuracy score](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score), [balanced accuracy score](https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score), and [F1-score](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics).

2) Or, install PyTEAP via `pip` as shown above, and use its modules and functions as you want.

The below table shows the results of baseline classification with `seed=0`.

|               | Gaussian NB | Random voting | Majority voting | Class ratio voting |
|--------------:|------------:|--------------:|----------------:|-------------------:|
| Acc.          |    0.508594 |      0.496094 |        0.553125 |           0.489063 |
| Balanced acc. |    0.513142 |      0.494228 |        0.500000 |           0.484621 |
| F1-score      |    0.513712 |      0.528224 |        0.712272 |           0.531533 |

\* You must have access to [DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/), in particular [a preprocessed data in Python format](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html) to perform above baseline classification. Please contact DEAP maintainers if you need access to the dataset.
