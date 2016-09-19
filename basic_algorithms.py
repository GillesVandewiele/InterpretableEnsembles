"""
Benchmark script for all basic algorithms/methods used on different datasets.

Algorithms:
-----------
    * C4.5
    * CART
    * QUEST
    * CN2Unordered
"""
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from constructors.c45orangeconstructor import C45Constructor
from constructors.cartconstructor import CARTConstructor
from constructors.cn2rulelearner import CN2UnorderedConstructor
from constructors.questconstructor import QuestConstructor
from data.load_all_datasets import load_all_datasets
from xgboost import XGBClassifier

import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt
import pylab as pl

# cart = CARTConstructor(max_depth=5, min_samples_leaf=2, criterion='gini')
# quest = QuestConstructor(max_nr_nodes=2, discrete_thresh=50, alpha=0.01)
# tree_constructors = [c45, cart, quest]

# cn2 = CN2UnorderedConstructor()
# rule_learners = [cn2]

rf = RandomForestClassifier(n_estimators=100, min_samples_split=2, criterion='gini')
ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.25)
ensembles = [rf, ada]

xgb = XGBClassifier()

datasets = load_all_datasets()

N_FOLDS = 3


def get_best_c45_classifier(train, label_col, skf_tune):
    c45 = C45Constructor()
    cfs = np.arange(0.05, 1.05, 0.05)
    cfs_errors = {}
    for cf in cfs:  cfs_errors[cf] = []

    for train_tune_idx, val_tune_idx in skf_tune:
        train_tune = train.iloc[train_tune_idx, :]
        X_train_tune = train_tune.drop(label_col, axis=1)
        y_train_tune = train_tune[label_col]
        val_tune = train.iloc[val_tune_idx, :]
        X_val_tune = val_tune.drop(label_col, axis=1)
        y_val_tune = val_tune[label_col]
        for cf in cfs:
            c45.cf = cf
            tree = c45.construct_tree(X_train_tune, y_train_tune)
            predictions = tree.evaluate_multiple(X_val_tune).astype(int)
            cfs_errors[cf].append(1 - accuracy_score(predictions, y_val_tune, normalize=True))

    for cf in cfs:
        cfs_errors[cf] = np.mean(cfs_errors[cf])

    c45.cf = min(cfs_errors.items(), key=operator.itemgetter(1))[0]
    return c45


def get_best_cart_classifier(train, label_col, skf_tune):
    cart = CARTConstructor()
    max_depths = np.arange(1,21,2)
    max_depths = np.append(max_depths, None)
    min_samples_splits = np.arange(1,20,1)

    errors = {}
    for max_depth in max_depths:
        for min_samples_split in min_samples_splits:
            errors[(max_depth, min_samples_split)] = []

    for train_tune_idx, val_tune_idx in skf_tune:
        train_tune = train.iloc[train_tune_idx, :]
        X_train_tune = train_tune.drop(label_col, axis=1)
        y_train_tune = train_tune[label_col]
        val_tune = train.iloc[val_tune_idx, :]
        X_val_tune = val_tune.drop(label_col, axis=1)
        y_val_tune = val_tune[label_col]
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                cart.max_depth = max_depth
                cart.min_samples_split = min_samples_split
                tree = cart.construct_tree(X_train_tune, y_train_tune)
                predictions = tree.evaluate_multiple(X_val_tune).astype(int)
                errors[((max_depth, min_samples_split))].append(1 - accuracy_score(predictions, y_val_tune, normalize=True))


    for max_depth in max_depths:
        for min_samples_split in min_samples_splits:
            errors[(max_depth, min_samples_split)] = np.mean(errors[(max_depth, min_samples_split)])

    best_params = min(errors.items(), key=operator.itemgetter(1))[0]
    cart.max_depth = best_params[0]
    cart.min_samples_split = best_params[1]

    return cart

def get_best_quest_classifier(train, label_col, skf_tune):
    quest = QuestConstructor()
    alphas = [10e-5, 10e-4, 10e-3, 10e-2, 0.25, 0.5, 0.9, 0.99]#np.arange(0.001, 1, 0.01)
    # max_nr_nodes = np.arange(1,20,2)

    errors = {}
    for alpha in alphas:
        # for max_nr_node in max_nr_nodes:
        errors[alpha] = []

    for train_tune_idx, val_tune_idx in skf_tune:
        train_tune = train.iloc[train_tune_idx, :]
        X_train_tune = train_tune.drop(label_col, axis=1)
        y_train_tune = train_tune[label_col]
        val_tune = train.iloc[val_tune_idx, :]
        X_val_tune = val_tune.drop(label_col, axis=1)
        y_val_tune = val_tune[label_col]
        for alpha in alphas:
            quest.alpha = alpha
            tree = quest.construct_tree(X_train_tune, y_train_tune)
            predictions = tree.evaluate_multiple(X_val_tune).astype(int)
            errors[alpha].append(1 - accuracy_score(predictions, y_val_tune, normalize=True))

    for alpha in alphas:
        # for max_nr_node in max_nr_nodes:
            errors[alpha] = np.mean(errors[alpha])

    best_params = min(errors.items(), key=operator.itemgetter(1))[0]
    quest.alpha = best_params

    return quest


def get_best_cn2_classifier(train, label_col, skf_tune):
    cn2 = CN2UnorderedConstructor()
    beam_widths = np.arange(1,20,3)
    # alphas = np.arange(0.1, 1, 0.2)
    alphas = [0.5]

    errors = {}
    for beam_width in beam_widths:
        for alpha in alphas:
            errors[(beam_width, alpha)] = []

    for train_tune_idx, val_tune_idx in skf_tune:
        train_tune = train.iloc[train_tune_idx, :]
        X_train_tune = train_tune.drop(label_col, axis=1)
        y_train_tune = train_tune[label_col]
        val_tune = train.iloc[val_tune_idx, :]
        X_val_tune = val_tune.drop(label_col, axis=1)
        y_val_tune = val_tune[label_col]
        for beam_width in beam_widths:
            for alpha in alphas:
                cn2.beam_width = beam_width
                cn2.alpha = alpha
                cn2.extract_rules(X_train_tune, y_train_tune)
                predictions = map(int, [prediction[0].value for prediction in cn2.classify(X_val_tune)])
                errors[(beam_width, alpha)].append(1 - accuracy_score(predictions, y_val_tune, normalize=True))
                print 1 - accuracy_score(predictions, y_val_tune, normalize=True), (beam_width, alpha)

    for beam_width in beam_widths:
        for alpha in alphas:
            errors[(beam_width, alpha)] = np.mean(errors[(beam_width, alpha)])

    best_params = min(errors.items(), key=operator.itemgetter(1))[0]
    cn2.beam_width = best_params[0]
    cn2.alpha = best_params[1]

    return cn2

for dataset in datasets:
    conf_matrices = {'C4.5': [], 'QUEST': [], 'CART': [], 'CN2': []}
    print 'DATASET', dataset['name']
    df = dataset['dataframe']
    label_col = dataset['label_col']
    feature_cols = dataset['feature_cols']
    skf = StratifiedKFold(df[label_col], n_folds=N_FOLDS, shuffle=True, random_state=1337)

    for train_idx, test_idx in skf:
        train = df.iloc[train_idx, :].reset_index(drop=True)
        X_train = train.drop(label_col, axis=1)
        y_train = train[label_col]
        test = df.iloc[test_idx, :].reset_index(drop=True)
        X_test = test.drop(label_col, axis=1)
        y_test = test[label_col]

        TUNE_FOLDS = 3
        skf_tune = StratifiedKFold(y_train.values, n_folds=TUNE_FOLDS, shuffle=True, random_state=1337)

        c45_clf = get_best_c45_classifier(train, label_col, skf_tune)
        print 'Best cf of C45', c45_clf.cf
        c45_tree = c45_clf.construct_tree(X_train, y_train)
        predictions = c45_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['C4.5'].append(confusion_matrix(predictions, y_test))

        cart_clf = get_best_cart_classifier(train, label_col, skf_tune)
        print 'Best (max_depth, min_samples_split) for CART', cart_clf.max_depth, cart_clf.min_samples_split
        cart_tree = cart_clf.construct_tree(X_train, y_train)
        predictions = cart_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['CART'].append(confusion_matrix(predictions, y_test))

        quest_clf = get_best_quest_classifier(train, label_col, skf_tune)
        print 'Best (alpha, max_nr_nodes) for QUEST', quest_clf.alpha, quest_clf.max_nr_nodes
        quest_tree = quest_clf.construct_tree(X_train, y_train)
        predictions = quest_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['QUEST'].append(confusion_matrix(predictions, y_test))

        cn2_clf = get_best_cn2_classifier(train, label_col, skf_tune)
        # cn2_clf = CN2UnorderedConstructor()
        print 'Best (alpha, beam_width) for CN2', cn2_clf.alpha, cn2_clf.beam_width
        cn2_clf.extract_rules(X_train, y_train)
        # cn2_clf.print_rules()
        # cn2_clf.rules_to_decision_space()
        predictions = map(int, [prediction[0].value for prediction in cn2_clf.classify(X_test)])
        conf_matrices['CN2'].append(confusion_matrix(predictions, y_test))

    fig = plt.figure()
    fig.suptitle('Accuracy on ' + dataset['name'] + ' dataset using ' + str(N_FOLDS) + ' folds', fontsize=20)
    counter = 0
    conf_matrices_mean = {}
    print conf_matrices
    for key in conf_matrices:
        conf_matrices_mean[key] = np.zeros(conf_matrices[key][0].shape)
        for i in range(len(conf_matrices[key])):
            conf_matrices_mean[key] = np.add(conf_matrices_mean[key], conf_matrices[key][i])
        cm_normalized = np.around(
            conf_matrices_mean[key].astype('float') / conf_matrices_mean[key].sum(axis=1)[:,
                                                                np.newaxis], 4)

        diagonal_sum = sum(
            [conf_matrices_mean[key][i][i] for i in range(len(conf_matrices_mean[key]))])
        total_count = np.sum(conf_matrices_mean[key])
        print conf_matrices_mean[key], float(diagonal_sum) / float(total_count)

        ax = fig.add_subplot(1, len(conf_matrices), counter + 1)
        cax = ax.matshow(cm_normalized, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
        ax.set_title(key, y=1.08)
        for (j, i), label in np.ndenumerate(cm_normalized):
            ax.text(i, j, label, ha='center', va='center')
        if counter == len(conf_matrices) - 1:
            fig.colorbar(cax, fraction=0.046, pad=0.04)
        counter += 1

    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 2, Size[1], forward=True)
    plt.show()
