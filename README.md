# PyTEAP

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

*PyTEAP* is a Python implementation of [Toolbox for Emotion Analysis using Physiological signals (TEAP)](https://github.com/Gijom/TEAP).

This project intends to reimplement TEAP, originally written in MATLAB, in Python, to enable interoperation with other Python packages.

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

## Baseline classification with DEAP
```console
$ python baseline.py --root '/path/to/deap_root'
```
Running [baseline.py](https://github.com/cheulyop/PyTEAP/blob/master/baseline.py) will load raw datafiles from the root directory, preprocess features and target labels, perform baseline classification using four simple classifiers: 1) [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html), 2) random voting assuming a uniform distribution between classes, 3) majority voting, 4) class ratio voting, and print classification results.

The below table shows the results of baseline classification with `seed=0`, with metrics of: [accuracy score](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score), [balanced accuracy score](https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score), and [F1-score](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics).

|               | Gaussian NB | Random voting | Majority voting | Class ratio voting |
|--------------:|------------:|--------------:|----------------:|-------------------:|
| Acc.          |    0.508594 |      0.496094 |        0.553125 |           0.489063 |
| Balanced acc. |    0.513142 |      0.494228 |        0.500000 |           0.484621 |
| F1-score      |    0.513712 |      0.528224 |        0.712272 |           0.531533 |

\* You must have access to  [DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/), in particular [a preprocessed data in Python format](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html) to perform above baseline classification. Please contact DEAP maintainers if you need access to the dataset.
