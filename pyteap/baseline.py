import os
import pickle
import logging
import argparse
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from numpy.random import default_rng
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from pyteap.signals.gsr import acquire_gsr, get_gsr_features
from pyteap.signals.bvp import acquire_bvp, get_bvp_features
from pyteap.signals.hst import acquire_hst, get_hst_features
from pyteap.utils.logging import init_logger


def prepare_deap(data_dir):
    X, y = {}, {}

    # for each datafile corresponding to a subject
    for fname in sorted(os.listdir(data_dir)):
        sbj_no = int(fname.split('.')[0][1:])
        fpath = os.path.join(data_dir, fname)

        # open file and get signals and labels
        with open(fpath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            signals, labels = data['data'], data['labels']

            # for each trial -- there are 40 trials for one subject
            curr_X, curr_y = [], []
            for i in tqdm(range(40), desc=f'Subject {sbj_no:02d}', ascii=True, dynamic_ncols=True):
                gsr = acquire_gsr(signals[i, 36, :], 128)
                bvp = acquire_bvp(signals[i, 38, :], 128)
                hst = acquire_hst(signals[i, 39, :], 128)

                # get features and labels
                gsr_features = get_gsr_features(gsr, 128)
                bvp_features = get_bvp_features(bvp, 128)
                hst_features = get_hst_features(hst, 128)
                features = np.concatenate([gsr_features, bvp_features, hst_features])
                targets = [int(labels[i][0] > 5), int(labels[i][1] > 5)]

                curr_X.append(features)
                curr_y.append(targets)
            
            # stack features for current subject and apply min-max scaling
            X[sbj_no] = StandardScaler().fit_transform(np.stack(curr_X))
            y[sbj_no] = np.stack(curr_y)

    features = np.concatenate(list(X.values()))
    targets = np.concatenate(list(y.values()))
    return features, targets


def pred_majority(majority, y_test):
    preds = np.repeat(majority, y_test.size)
    res = {
        'acc.': accuracy_score(y_test, preds),
        'bacc.': balanced_accuracy_score(y_test, preds, adjusted=False),
        'f1': f1_score(y_test, preds)
    }
    return res


def pred_random(y_classes, y_test, seed, rng, ratios=None):
    preds = rng.choice(y_classes, y_test.size, replace=True, p=ratios)
    res = {
        'acc.': accuracy_score(y_test, preds),
        'bacc.': balanced_accuracy_score(y_test, preds, adjusted=False),
        'f1': f1_score(y_test, preds)
    }
    return res


def pred_GaussianNB(X_train, y_train, X_test, y_test):
    clf = GaussianNB()
    preds = clf.fit(X_train, y_train).predict(X_test)
    res = {
        'acc.': clf.score(X_test, y_test),
        'bacc.': balanced_accuracy_score(y_test, preds, adjusted=False),
        'f1': f1_score(y_test, preds)
    }
    return res


def get_baseline_kfold(X, y, seed, target_class='valence', n_splits=5, shuffle=True):
    # get labels corresponding to target class
    if target_class == 'valence':
        y = y[:, 0]
    elif target_class == 'arousal':
        y = y[:, 1]

    # initialize random number generator and fold generator
    rng = default_rng(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    results = {}
    # for each fold, split train & test and get classification results
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_classes, y_counts = np.unique(y_train, return_counts=True)
        majority = y_classes[np.argmax(y_counts)]
        class_ratios = y_counts / y_train.size

        results[i+1] = {
            'Gaussian NB': pred_GaussianNB(X_train, y_train, X_test, y_test),
            'Random': pred_random(y_classes, y_test, seed, rng),
            'Majority': pred_majority(majority, y_test),
            'Class ratio': pred_random(y_classes, y_test, seed, rng, ratios=class_ratios)
        }

    # return results as table
    results = {(fold, classifier): values for (fold, _results) in results.items() for (classifier, values) in _results.items()}
    results_table = pd.DataFrame.from_dict(results, orient='index').stack().unstack(level=1).rename_axis(['Fold', 'Metric'])
    return results_table


if __name__ == "__main__":
    # initialize parser
    parser = argparse.ArgumentParser(description='Preprocess DEAP dataset and get baseline classification results.')
    parser.add_argument('--root', '-r', type=str, required=True, help='path to the dataset directory')
    parser.add_argument('--seed', '-s', type=int, default=0, help='set seed for random number generation')
    args = parser.parse_args()

    # initialize default logger
    logger = init_logger()

    # filter these numpy RuntimeWarning messages
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')

    # get features and targets from raw data
    data_dir = os.path.expanduser(args.root)
    logger.info(f'Processing trials from {data_dir}...')
    features, targets = prepare_deap(data_dir)
    logger.info('Processing complete.')

    # get classification results
    results = get_baseline_kfold(features, targets, args.seed, target_class='valence')

    # print summary of classification results
    print(results.groupby(level='Metric').mean())
