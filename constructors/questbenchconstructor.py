"""
    Written by Kiani Lannoye & Gilles Vandewiele
    Commissioned by UGent.

    Design of a diagnose- and follow-up platform for patients with chronic headaches
"""

import Orange
import time

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
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


class QUESTBenchConstructor(TreeConstructor):
    """
    This class contains a wrapper around an implementation of GUIDE, written by Loh.
    http://www.stat.wisc.edu/~loh/guide.html
    """

    def __init__(self):
        pass

    def get_name(self):
        return "QUESTLoh"

    def construct_tree(self, training_feature_vectors, labels):
        self.create_desc_and_data_file(training_feature_vectors, labels)
        input = open("in.txt", "w")
        output = file('out.txt', 'w')
        p = subprocess.Popen(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1])+'/quest > log.txt', stdin=subprocess.PIPE, shell=True)
        p.stdin.write("2\n")
        p.stdin.write("in.txt\n")
        p.stdin.write("1\n")
        p.stdin.write("out.txt\n")
        p.stdin.write("1\n")
        p.stdin.write("dsc.txt\n")
        p.stdin.write("1\n")
        p.stdin.write("\n")
        p.wait()
        input.close()
        output.close()

        while not os.path.exists('in.txt'):
            time.sleep(1)
        p = subprocess.Popen(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1])+'/quest < in.txt > log.txt', stdin=subprocess.PIPE, shell=True)
        p.wait()

        output = file('out.txt', 'r')
        lines = output.readlines()
        output.close()

        start_index, end_index, counter = 0, 0, 0
        for line in lines:
            if line == '  Classification tree:\n':
                start_index = counter+2
            if line == '  Information for each node:\n':
                end_index = counter-1
            counter += 1
        tree = self.decision_tree_from_text(lines[start_index:end_index])

        self.remove_files()

        # tree.visualise('QUEST')
        return tree

    def decision_tree_from_text(self, lines):

        dt = DecisionTree()

        if '<=' in lines[0] or '>' in lines[0]:
            # Intermediate node
            node_name = lines[0].split(':')[0].lstrip()
            label, value = lines[0].split(':')[1].split('<=')
            label = ' '.join(label.lstrip().rstrip().split('.'))
            value = value.lstrip().split()[0]
            dt.label = label
            dt.value = float(value)
            dt.left = self.decision_tree_from_text(lines[1:])
            counter = 1
            while lines[counter].split(':')[0].lstrip() != node_name: counter+=1
            dt.right = self.decision_tree_from_text(lines[counter+1:])
        else:
            # Terminal node
            dt.label = int(eval(lines[0].split(':')[1].lstrip()))

        return dt

    def create_desc_and_data_file(self, training_feature_vectors, labels):
        dsc = open("dsc.txt", "w")
        data = open("data.txt", "w")

        dsc.write("data.txt\n")
        dsc.write("\"?\"\n")
        dsc.write("column, var, type\n")
        count = 1
        for col in training_feature_vectors.columns:
            dsc.write(str(count) + ' \"' + str(col) + '\" n\n')
            count += 1
        dsc.write(str(count) + ' ' + str(labels.name) + ' d')

        for i in range(len(training_feature_vectors)):
            sample = training_feature_vectors.iloc[i,:]
            for col in training_feature_vectors.columns:
                data.write(str(sample[col]) + ' ')
            if i != len(training_feature_vectors)-1:
                data.write(str(labels[i])+'\n')
            else:
                data.write(str(labels[i]))

        data.close()
        dsc.close()

    def remove_files(self):
        os.remove('data.txt')
        os.remove('in.txt')
        os.remove('dsc.txt')
        os.remove('out.txt')
        os.remove('log.txt')


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

def get_best_cn2_classifier(train, label_col, skf_tune):
    cn2 = CN2UnorderedConstructor()
    beam_widths = np.arange(1,20,3)
    # alphas = np.arange(0.1, 1, 0.2)
    alphas = [0.25, 0.5, 0.75]

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
                # print 1 - accuracy_score(predictions, y_val_tune, normalize=True), (beam_width, alpha)

    for beam_width in beam_widths:
        for alpha in alphas:
            errors[(beam_width, alpha)] = np.mean(errors[(beam_width, alpha)])

    best_params = min(errors.items(), key=operator.itemgetter(1))[0]
    cn2.beam_width = best_params[0]
    cn2.alpha = best_params[1]

    return cn2

# datasets = load_all_datasets()
# quest_bench = QUESTBenchConstructor()
# guide = GUIDEConstructor()
# quest = QuestConstructor()
# merger = DecisionTreeMerger()
# NR_FOLDS = 3
# for dataset in datasets:
#     print dataset['name'], len(dataset['dataframe'])
#     conf_matrices = {'QUESTGilles': [], 'GUIDE': [], 'C4.5': [], 'CART': [], 'ISM': [], 'ISM_pruned': [],
#                      'Genetic': []}  #
#     avg_nodes = {'QUESTGilles': [], 'GUIDE': [], 'C4.5': [], 'CART': [], 'ISM': [], 'ISM_pruned': [],
#                  'Genetic': []}  #
#     df = dataset['dataframe']
#     label_col = dataset['label_col']
#     feature_cols = dataset['feature_cols']
#     skf = StratifiedKFold(df[label_col], n_folds=NR_FOLDS, shuffle=True, random_state=1337)
#
#     for fold, (train_idx, test_idx) in enumerate(skf):
#         print 'Fold', fold+1, '/', NR_FOLDS
#         train = df.iloc[train_idx, :].reset_index(drop=True)
#         X_train = train.drop(label_col, axis=1)
#         y_train = train[label_col]
#         test = df.iloc[test_idx, :].reset_index(drop=True)
#         X_test = test.drop(label_col, axis=1)
#         y_test = test[label_col]
#
#         smote = SMOTE(ratio='auto', kind='regular')
#         print len(X_train)
#         X_train, y_train = smote.fit_sample(X_train, y_train)
#         X_train = DataFrame(X_train, columns=feature_cols)
#         y_train = DataFrame(y_train, columns=[label_col])[label_col]
#         perm = np.random.permutation(len(X_train))
#         X_train = X_train.iloc[perm].reset_index(drop=True)
#         y_train = y_train.iloc[perm].reset_index(drop=True)
#         train = X_train.copy()
#         train[y_train.name] = Series(y_train, index=train.index)
#         print len(X_train)
#
#         # quest_bench_tree = quest_bench.construct_tree(X_train, y_train)
#         # predictions = quest_bench_tree.evaluate_multiple(X_test).astype(int)
#         # conf_matrices['QUESTLoh'].append(confusion_matrix(y_test, predictions))
#         # avg_nodes['QUESTLoh'].append(quest_bench_tree.count_nodes())
#
#         print 'QUEST'
#         quest_tree = quest.construct_tree(X_train, y_train)
#         predictions = quest_tree.evaluate_multiple(X_test).astype(int)
#         conf_matrices['QUESTGilles'].append(confusion_matrix(y_test, predictions))
#         avg_nodes['QUESTGilles'].append(quest_tree.count_nodes())
#
#         print 'GUIDE'
#         guide_tree = guide.construct_tree(X_train, y_train)
#         predictions = guide_tree.evaluate_multiple(X_test).astype(int)
#         conf_matrices['GUIDE'].append(confusion_matrix(y_test, predictions))
#         avg_nodes['GUIDE'].append(guide_tree.count_nodes())
#
#         skf_tune = StratifiedKFold(train[label_col], n_folds=3, shuffle=True, random_state=1337)
#
#         print 'C4.5'
#         c45_clf = get_best_c45_classifier(train, label_col, skf_tune)
#         c45_tree = c45_clf.construct_tree(X_train, y_train)
#         predictions = c45_tree.evaluate_multiple(X_test).astype(int)
#         conf_matrices['C4.5'].append(confusion_matrix(y_test, predictions))
#         avg_nodes['C4.5'].append(c45_tree.count_nodes())
#
#         print 'CART'
#         cart_clf = get_best_cart_classifier(train, label_col, skf_tune)
#         cart_tree = cart_clf.construct_tree(X_train, y_train)
#         predictions = cart_tree.evaluate_multiple(X_test).astype(int)
#         conf_matrices['CART'].append(confusion_matrix(y_test, predictions))
#         avg_nodes['CART'].append(cart_tree.count_nodes())
#
#         # print 'CN2'
#         # cn2_clf = get_best_cn2_classifier(train, label_col, skf_tune)
#         # cn2 = cn2_clf.extract_rules(X_train, y_train)
#         # predictions = map(int, [prediction[0].value for prediction in cn2_clf.classify(X_test)])
#         # conf_matrices['CN2'].append(confusion_matrix(y_test, predictions))
#         # avg_nodes['CN2'].append(len(cn2_clf.model.rules))
#
#         print 'Got all trees, lets merge them!'
#         trees = [quest_tree, guide_tree, c45_tree, cart_tree] # quest_bench_tree
#
#         predictions_list = []
#         for tree in trees:
#             tree.data = train
#             tree.populate_samples(X_train, y_train.astype(str).values)
#             predictions_list.append(tree.evaluate_multiple(X_test).astype(int))
#         print 'Correlation of no bootstrapping: ', np.corrcoef(predictions_list)
#
#         constructors = [c45_clf, cart_clf, quest, guide]
#
#         for tree in bootstrap(train, label_col, constructors, boosting=True, nr_classifiers=3):
#             tree.data = train
#             tree.populate_samples(X_train, y_train.astype(str).values)
#             predictions_list.append(tree.evaluate_multiple(X_test).astype(int))
#         print 'Correlation of bootstrapping: ', np.corrcoef(predictions_list)
#
#
#         train_gen = train.rename(columns={'Class':'cat'})
#         genetic = merger.genetic_algorithm(train_gen, 'cat', constructors, seed=1337, num_iterations=5,
#                                            num_mutations=12, population_size=25, max_samples=1, val_fraction=0.1,
#                                            num_boosts=1)
#         predictions = genetic.evaluate_multiple(X_test).astype(int)
#         conf_matrices['Genetic'].append(confusion_matrix(y_test, predictions))
#         avg_nodes['Genetic'].append(genetic.count_nodes())
#
#         # genetic_prune = merger.genetic_algorithm(train_gen, 'cat', constructors, seed=1337, num_iterations=5,
#         #                                    num_mutations=12, population_size=25, max_samples=1, val_fraction=0.1,
#         #                                    num_boosts=1, prune=True)
#         # predictions = genetic_prune.evaluate_multiple(X_test).astype(int)
#         # conf_matrices['Genetic_prune'].append(confusion_matrix(y_test, predictions))
#         # avg_nodes['Genetic_prune'].append(genetic_prune.count_nodes())
#
#         ism_tree = ism(bootstrap(train, label_col, constructors, boosting=True, nr_classifiers=3), train, label_col,
#                        min_nr_samples=1, calc_fracs_from_ensemble=True)
#         predictions = ism_tree.evaluate_multiple(X_test).astype(int)
#         conf_matrices['ISM'].append(confusion_matrix(y_test, predictions))
#         avg_nodes['ISM'].append(ism_tree.count_nodes())
#
#         print 'Lets prune the tree'
#         ism_pruned = ism_tree.cost_complexity_pruning(X_train, y_train, 'ism', ism_constructors=constructors,
#                                                       ism_calc_fracs=True, n_folds=3, ism_nr_classifiers=3,
#                                                       ism_boosting=True)
#         predictions = ism_pruned.evaluate_multiple(X_test).astype(int)
#         conf_matrices['ISM_pruned'].append(confusion_matrix(y_test, predictions))
#         print conf_matrices['ISM'][len(conf_matrices['ISM'])-1]
#         print conf_matrices['ISM_pruned'][len(conf_matrices['ISM_pruned'])-1]
#         avg_nodes['ISM_pruned'].append(ism_pruned.count_nodes())
#
#     fig = plt.figure()
#     fig.suptitle('Accuracy on ' + dataset['name'] + ' dataset using ' + str(5) + ' folds', fontsize=20)
#     counter = 0
#     conf_matrices_mean = {}
#     print conf_matrices
#     for key in conf_matrices:
#         conf_matrices_mean[key] = np.zeros(conf_matrices[key][0].shape)
#         for i in range(len(conf_matrices[key])):
#             conf_matrices_mean[key] = np.add(conf_matrices_mean[key], conf_matrices[key][i])
#         cm_normalized = np.around(
#             conf_matrices_mean[key].astype('float') / conf_matrices_mean[key].sum(axis=1)[:,
#                                                       np.newaxis], 4)
#
#         diagonal_sum = sum(
#             [conf_matrices_mean[key][i][i] for i in range(len(conf_matrices_mean[key]))])
#         norm_diagonal_sum = sum(
#             [conf_matrices_mean[key][i][i]/sum(conf_matrices_mean[key][i]) for i in range(len(conf_matrices_mean[key]))])
#         total_count = np.sum(conf_matrices_mean[key])
#         print conf_matrices_mean[key], float(diagonal_sum) / float(total_count)
#         print 'Balanced accuracy: ', float(norm_diagonal_sum) / conf_matrices_mean[key].shape[0]
#
#         ax = fig.add_subplot(2, np.math.ceil(len(conf_matrices) / 2.0), counter + 1)
#         cax = ax.matshow(cm_normalized, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
#         ax.set_title(key + '(' + str(sum(avg_nodes[key])/len(avg_nodes[key])) + ')', y=1.08)
#         for (j, i), label in np.ndenumerate(cm_normalized):
#             ax.text(i, j, label, ha='center', va='center')
#         if counter == len(conf_matrices) - 1:
#             fig.colorbar(cax, fraction=0.046, pad=0.04)
#         counter += 1
#
#     F = plt.gcf()
#     Size = F.get_size_inches()
#     F.set_size_inches(Size[0] * 2, Size[1], forward=True)
#     plt.show()
