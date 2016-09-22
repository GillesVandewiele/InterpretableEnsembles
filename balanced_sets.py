import Orange
import time

from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import TomekLinks
from pandas import DataFrame, Series
import os
import subprocess

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from ISM_v3 import ism, bootstrap
from constructors.c45orangeconstructor import C45Constructor
from constructors.cartconstructor import CARTConstructor
from constructors.cn2rulelearner import CN2UnorderedConstructor
from constructors.guideconstructor import GUIDEConstructor
from constructors.questconstructor import QuestConstructor
from constructors.treeconstructor import TreeConstructor
from constructors.treemerger import DecisionTreeMerger
from data.load_all_datasets import load_all_datasets
from decisiontree import DecisionTree
from pandas_to_orange import df2table
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import operator


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

datasets = load_all_datasets()
NR_FOLDS = 5
for dataset in datasets:
    print dataset['name']
    conf_matrices = {'Imbalanced': [], 'Random US': [], 'Tomek': [], 'Cluster': [], 'Instance NN': [], 'Edited NN': [],
                     'Random OS': [], 'SMOTE': [], 'SMOTE SVM': [], 'SMOTE Tomek': [], 'SMOTE ENN': []}
    avg_nodes = {'Imbalanced': [], 'Random US': [], 'Tomek': [], 'Cluster': [], 'Instance NN': [], 'Edited NN': [],
                 'Random OS': [], 'SMOTE': [], 'SMOTE SVM': [], 'SMOTE Tomek': [], 'SMOTE ENN': []}
    df = dataset['dataframe']
    label_col = dataset['label_col']
    feature_cols = dataset['feature_cols']
    skf = StratifiedKFold(df[label_col], n_folds=NR_FOLDS, shuffle=True, random_state=1337)

    for fold, (train_idx, test_idx) in enumerate(skf):
        print 'Fold', fold+1, '/', NR_FOLDS
        train = df.iloc[train_idx, :].reset_index(drop=True)
        X_train = train.drop(label_col, axis=1)
        y_train = train[label_col]
        test = df.iloc[test_idx, :].reset_index(drop=True)
        X_test = test.drop(label_col, axis=1)
        y_test = test[label_col]

        # smote = SMOTE(ratio='auto', kind='regular')
        # STK = SMOTETomek(ratio='auto')
        # print len(X_train)
        # X_train, y_train = STK.fit_sample(X_train, y_train)
        # X_train = DataFrame(X_train, columns=feature_cols)
        # y_train = DataFrame(y_train, columns=[label_col])[label_col]
        # perm = np.random.permutation(len(X_train))
        # X_train = X_train.iloc[perm].reset_index(drop=True)
        # y_train = y_train.iloc[perm].reset_index(drop=True)
        # train = X_train.copy()
        # train[y_train.name] = Series(y_train, index=train.index)
        # print len(X_train)

        skf_tune = StratifiedKFold(train[label_col], n_folds=3, shuffle=True, random_state=1337)

        c45_clf = get_best_c45_classifier(train, label_col, skf_tune)
        c45_tree = c45_clf.construct_tree(X_train, y_train)
        predictions = c45_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['Imbalanced'].append(confusion_matrix(y_test, predictions))
        avg_nodes['Imbalanced'].append(c45_tree.count_nodes())

        print 'Random Undersampling'
        US = RandomUnderSampler()
        prior_nodes = len(X_train)
        X_train_sampled, y_train_sampled = US.fit_sample(X_train, y_train)
        X_train_sampled = DataFrame(X_train_sampled, columns=feature_cols)
        y_train_sampled = DataFrame(y_train_sampled, columns=[label_col])[label_col]
        perm = np.random.permutation(len(X_train_sampled))
        X_train_sampled = X_train_sampled.iloc[perm].reset_index(drop=True)
        y_train_sampled = y_train_sampled.iloc[perm].reset_index(drop=True)
        train = X_train_sampled.copy()
        train[y_train_sampled.name] = Series(y_train_sampled, index=train.index)
        print 'From', prior_nodes, 'to', len(X_train_sampled)

        skf_tune = StratifiedKFold(train[label_col], n_folds=3, shuffle=True, random_state=1337)

        c45_clf = get_best_c45_classifier(train, label_col, skf_tune)
        c45_tree = c45_clf.construct_tree(X_train_sampled, y_train_sampled)
        predictions = c45_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['Random US'].append(confusion_matrix(y_test, predictions))
        avg_nodes['Random US'].append(c45_tree.count_nodes())

        print 'Tomek'
        TK = TomekLinks()
        prior_nodes = len(X_train)
        X_train_sampled, y_train_sampled = TK.fit_sample(X_train, y_train)
        X_train_sampled = DataFrame(X_train_sampled, columns=feature_cols)
        y_train_sampled = DataFrame(y_train_sampled, columns=[label_col])[label_col]
        perm = np.random.permutation(len(X_train_sampled))
        X_train_sampled = X_train_sampled.iloc[perm].reset_index(drop=True)
        y_train_sampled = y_train_sampled.iloc[perm].reset_index(drop=True)
        train = X_train_sampled.copy()
        train[y_train_sampled.name] = Series(y_train_sampled, index=train.index)
        print 'From', prior_nodes, 'to', len(X_train_sampled)

        skf_tune = StratifiedKFold(train[label_col], n_folds=3, shuffle=True, random_state=1337)

        c45_clf = get_best_c45_classifier(train, label_col, skf_tune)
        c45_tree = c45_clf.construct_tree(X_train_sampled, y_train_sampled)
        predictions = c45_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['Tomek'].append(confusion_matrix(y_test, predictions))
        avg_nodes['Tomek'].append(c45_tree.count_nodes())

        print 'Clustering'
        CC = ClusterCentroids()
        prior_nodes = len(X_train)
        X_train_sampled, y_train_sampled = CC.fit_sample(X_train, y_train)
        X_train_sampled = DataFrame(X_train_sampled, columns=feature_cols)
        y_train_sampled = DataFrame(y_train_sampled, columns=[label_col])[label_col]
        perm = np.random.permutation(len(X_train_sampled))
        X_train_sampled = X_train_sampled.iloc[perm].reset_index(drop=True)
        y_train_sampled = y_train_sampled.iloc[perm].reset_index(drop=True)
        train = X_train_sampled.copy()
        train[y_train_sampled.name] = Series(y_train_sampled, index=train.index)
        print 'From', prior_nodes, 'to', len(X_train_sampled)

        skf_tune = StratifiedKFold(train[label_col], n_folds=3, shuffle=True, random_state=1337)

        c45_clf = get_best_c45_classifier(train, label_col, skf_tune)
        c45_tree = c45_clf.construct_tree(X_train_sampled, y_train_sampled)
        predictions = c45_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['Cluster'].append(confusion_matrix(y_test, predictions))
        avg_nodes['Cluster'].append(c45_tree.count_nodes())

        print 'Instance NN'
        IHT = InstanceHardnessThreshold()
        prior_nodes = len(X_train)
        X_train_sampled, y_train_sampled = IHT.fit_sample(X_train, y_train)
        X_train_sampled = DataFrame(X_train_sampled, columns=feature_cols)
        y_train_sampled = DataFrame(y_train_sampled, columns=[label_col])[label_col]
        perm = np.random.permutation(len(X_train_sampled))
        X_train_sampled = X_train_sampled.iloc[perm].reset_index(drop=True)
        y_train_sampled = y_train_sampled.iloc[perm].reset_index(drop=True)
        train = X_train_sampled.copy()
        train[y_train_sampled.name] = Series(y_train_sampled, index=train.index)
        print 'From', prior_nodes, 'to', len(X_train_sampled)

        skf_tune = StratifiedKFold(train[label_col], n_folds=3, shuffle=True, random_state=1337)

        c45_clf = get_best_c45_classifier(train, label_col, skf_tune)
        c45_tree = c45_clf.construct_tree(X_train_sampled, y_train_sampled)
        predictions = c45_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['Instance NN'].append(confusion_matrix(y_test, predictions))
        avg_nodes['Instance NN'].append(c45_tree.count_nodes())

        print 'Edited NN'
        RENN = RepeatedEditedNearestNeighbours()
        prior_nodes = len(X_train)
        X_train_sampled, y_train_sampled = RENN.fit_sample(X_train, y_train)
        X_train_sampled = DataFrame(X_train_sampled, columns=feature_cols)
        y_train_sampled = DataFrame(y_train_sampled, columns=[label_col])[label_col]
        perm = np.random.permutation(len(X_train_sampled))
        X_train_sampled = X_train_sampled.iloc[perm].reset_index(drop=True)
        y_train_sampled = y_train_sampled.iloc[perm].reset_index(drop=True)
        train = X_train_sampled.copy()
        train[y_train_sampled.name] = Series(y_train_sampled, index=train.index)
        print 'From', prior_nodes, 'to', len(X_train_sampled)

        skf_tune = StratifiedKFold(train[label_col], n_folds=3, shuffle=True, random_state=1337)

        c45_clf = get_best_c45_classifier(train, label_col, skf_tune)
        c45_tree = c45_clf.construct_tree(X_train_sampled, y_train_sampled)
        predictions = c45_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['Edited NN'].append(confusion_matrix(y_test, predictions))
        avg_nodes['Edited NN'].append(c45_tree.count_nodes())

        print 'Random OS'
        OS = RandomOverSampler(ratio='auto')
        prior_nodes = len(X_train)
        X_train_sampled, y_train_sampled = OS.fit_sample(X_train, y_train)
        X_train_sampled = DataFrame(X_train_sampled, columns=feature_cols)
        y_train_sampled = DataFrame(y_train_sampled, columns=[label_col])[label_col]
        perm = np.random.permutation(len(X_train_sampled))
        X_train_sampled = X_train_sampled.iloc[perm].reset_index(drop=True)
        y_train_sampled = y_train_sampled.iloc[perm].reset_index(drop=True)
        train = X_train_sampled.copy()
        train[y_train_sampled.name] = Series(y_train_sampled, index=train.index)
        print 'From', prior_nodes, 'to', len(X_train_sampled)

        skf_tune = StratifiedKFold(train[label_col], n_folds=3, shuffle=True, random_state=1337)

        c45_clf = get_best_c45_classifier(train, label_col, skf_tune)
        c45_tree = c45_clf.construct_tree(X_train_sampled, y_train_sampled)
        predictions = c45_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['Random OS'].append(confusion_matrix(y_test, predictions))
        avg_nodes['Random OS'].append(c45_tree.count_nodes())

        print 'SMOTE'
        smote = SMOTE(ratio='auto', kind='regular')
        prior_nodes = len(X_train)
        X_train_sampled, y_train_sampled = smote.fit_sample(X_train, y_train)
        X_train_sampled = DataFrame(X_train_sampled, columns=feature_cols)
        y_train_sampled = DataFrame(y_train_sampled, columns=[label_col])[label_col]
        perm = np.random.permutation(len(X_train_sampled))
        X_train_sampled = X_train_sampled.iloc[perm].reset_index(drop=True)
        y_train_sampled = y_train_sampled.iloc[perm].reset_index(drop=True)
        train = X_train_sampled.copy()
        train[y_train_sampled.name] = Series(y_train_sampled, index=train.index)
        print 'From', prior_nodes, 'to', len(X_train_sampled)

        skf_tune = StratifiedKFold(train[label_col], n_folds=3, shuffle=True, random_state=1337)

        c45_clf = get_best_c45_classifier(train, label_col, skf_tune)
        c45_tree = c45_clf.construct_tree(X_train_sampled, y_train_sampled)
        predictions = c45_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['SMOTE'].append(confusion_matrix(y_test, predictions))
        avg_nodes['SMOTE'].append(c45_tree.count_nodes())

        print 'SMOTE SVM'
        svm_args={'class_weight': 'auto'}
        svmsmote = SMOTE(ratio='auto', kind='svm', **svm_args)
        prior_nodes = len(X_train)
        X_train_sampled, y_train_sampled = svmsmote.fit_sample(X_train, y_train)
        X_train_sampled = DataFrame(X_train_sampled, columns=feature_cols)
        y_train_sampled = DataFrame(y_train_sampled, columns=[label_col])[label_col]
        perm = np.random.permutation(len(X_train_sampled))
        X_train_sampled = X_train_sampled.iloc[perm].reset_index(drop=True)
        y_train_sampled = y_train_sampled.iloc[perm].reset_index(drop=True)
        train = X_train_sampled.copy()
        train[y_train_sampled.name] = Series(y_train_sampled, index=train.index)
        print 'From', prior_nodes, 'to', len(X_train_sampled)

        skf_tune = StratifiedKFold(train[label_col], n_folds=3, shuffle=True, random_state=1337)

        c45_clf = get_best_c45_classifier(train, label_col, skf_tune)
        c45_tree = c45_clf.construct_tree(X_train_sampled, y_train_sampled)
        predictions = c45_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['SMOTE SVM'].append(confusion_matrix(y_test, predictions))
        avg_nodes['SMOTE SVM'].append(c45_tree.count_nodes())

        print 'SMOTE Tomek'
        STK = SMOTETomek(ratio='auto')
        prior_nodes = len(X_train)
        X_train_sampled, y_train_sampled = STK.fit_sample(X_train, y_train)
        X_train_sampled = DataFrame(X_train_sampled, columns=feature_cols)
        y_train_sampled = DataFrame(y_train_sampled, columns=[label_col])[label_col]
        perm = np.random.permutation(len(X_train_sampled))
        X_train_sampled = X_train_sampled.iloc[perm].reset_index(drop=True)
        y_train_sampled = y_train_sampled.iloc[perm].reset_index(drop=True)
        train = X_train_sampled.copy()
        train[y_train_sampled.name] = Series(y_train_sampled, index=train.index)
        print 'From', prior_nodes, 'to', len(X_train_sampled)

        skf_tune = StratifiedKFold(train[label_col], n_folds=3, shuffle=True, random_state=1337)

        c45_clf = get_best_c45_classifier(train, label_col, skf_tune)
        c45_tree = c45_clf.construct_tree(X_train_sampled, y_train_sampled)
        predictions = c45_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['SMOTE Tomek'].append(confusion_matrix(y_test, predictions))
        avg_nodes['SMOTE Tomek'].append(c45_tree.count_nodes())

        print 'SMOTE ENN'
        SENN = SMOTEENN(ratio='auto')
        prior_nodes = len(X_train)
        X_train_sampled, y_train_sampled = SENN.fit_sample(X_train, y_train)
        X_train_sampled = DataFrame(X_train_sampled, columns=feature_cols)
        y_train_sampled = DataFrame(y_train_sampled, columns=[label_col])[label_col]
        perm = np.random.permutation(len(X_train_sampled))
        X_train_sampled = X_train_sampled.iloc[perm].reset_index(drop=True)
        y_train_sampled = y_train_sampled.iloc[perm].reset_index(drop=True)
        train = X_train_sampled.copy()
        train[y_train_sampled.name] = Series(y_train_sampled, index=train.index)
        print 'From', prior_nodes, 'to', len(X_train_sampled)

        skf_tune = StratifiedKFold(train[label_col], n_folds=3, shuffle=True, random_state=1337)

        c45_clf = get_best_c45_classifier(train, label_col, skf_tune)
        c45_tree = c45_clf.construct_tree(X_train_sampled, y_train_sampled)
        predictions = c45_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['SMOTE ENN'].append(confusion_matrix(y_test, predictions))
        avg_nodes['SMOTE ENN'].append(c45_tree.count_nodes())

    fig = plt.figure()
    fig.suptitle('Accuracy on ' + dataset['name'] + ' dataset using ' + str(5) + ' folds', fontsize=20)
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
        norm_diagonal_sum = sum(
            [conf_matrices_mean[key][i][i]/sum(conf_matrices_mean[key][i]) for i in range(len(conf_matrices_mean[key]))])
        total_count = np.sum(conf_matrices_mean[key])
        print key
        print conf_matrices_mean[key], float(diagonal_sum) / float(total_count)
        print 'Balanced accuracy: ', float(norm_diagonal_sum) / conf_matrices_mean[key].shape[0]

        ax = fig.add_subplot(1, len(conf_matrices), counter + 1)
        cax = ax.matshow(cm_normalized, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
        ax.set_title(key + '(' + str(sum(avg_nodes[key])/len(avg_nodes[key])) + ')', y=1.08)
        for (j, i), label in np.ndenumerate(cm_normalized):
            ax.text(i, j, label, ha='center', va='center')
        if counter == len(conf_matrices) - 1:
            fig.colorbar(cax, fraction=0.046, pad=0.04)
        counter += 1

    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 2, Size[1], forward=True)
    plt.show()