"""
    Written by Kiani Lannoye & Gilles Vandewiele
    Commissioned by UGent.

    Design of a diagnose- and follow-up platform for patients with chronic headaches
"""

import Orange
import time
from pandas import DataFrame
import os
import subprocess

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix

from constructors.guideconstructor import GUIDEConstructor
from constructors.questconstructor import QuestConstructor
from constructors.treeconstructor import TreeConstructor
from data.load_all_datasets import load_all_datasets
from decisiontree import DecisionTree
from pandas_to_orange import df2table
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np


class QUESTBenchConstructor(TreeConstructor):
    """
    This class contains a wrapper around an implementation of GUIDE, written by Loh.
    http://www.stat.wisc.edu/~loh/guide.html
    """

    def __init__(self):
        pass

    def get_name(self):
        return "GUIDE"

    def construct_tree(self, training_feature_vectors, labels):
        self.create_desc_and_data_file(training_feature_vectors, labels)
        input = open("in.txt", "w")
        output = file('out.txt', 'w')
        p = subprocess.Popen('./quest > log.txt', stdin=subprocess.PIPE, shell=True)
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
        p = subprocess.Popen('./quest < in.txt > log.txt', stdin=subprocess.PIPE, shell=True)
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
            dt.label = int(lines[0].split(':')[1].lstrip())

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


datasets = load_all_datasets()
quest_bench = QUESTBenchConstructor()
guide = GUIDEConstructor()
quest = QuestConstructor()
for dataset in datasets:
    conf_matrices = {'QUESTGilles': [], 'QUESTLoh': [], 'GUIDE': []}
    df = dataset['dataframe']
    label_col = dataset['label_col']
    feature_cols = dataset['feature_cols']
    skf = StratifiedKFold(df[label_col], n_folds=3, shuffle=True, random_state=1337)

    for train_idx, test_idx in skf:
        train = df.iloc[train_idx, :].reset_index(drop=True)
        X_train = train.drop(label_col, axis=1)
        y_train = train[label_col]
        test = df.iloc[test_idx, :].reset_index(drop=True)
        X_test = test.drop(label_col, axis=1)
        y_test = test[label_col]

        quest_bench_tree = quest_bench.construct_tree(X_train, y_train)
        predictions = quest_bench_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['QUESTLoh'].append(confusion_matrix(predictions, y_test))

        quest_tree = quest.construct_tree(X_train, y_train)
        predictions = quest_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['QUESTGilles'].append(confusion_matrix(predictions, y_test))

        guide_tree = guide.construct_tree(X_train, y_train)
        predictions = guide_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['GUIDE'].append(confusion_matrix(predictions, y_test))

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
