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

from constructors.treeconstructor import TreeConstructor
from data.load_all_datasets import load_all_datasets
from decisiontree import DecisionTree
from pandas_to_orange import df2table


class GUIDEConstructor(TreeConstructor):
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
        p = subprocess.Popen('./guide', stdin=subprocess.PIPE, shell=True)
        p.stdin.write("1\n")
        p.stdin.write("in.txt\n")
        p.stdin.write("1\n")
        p.stdin.write("1\n")
        p.stdin.write("out.txt\n")
        p.stdin.write("1\n")
        p.stdin.write("1\n")
        p.stdin.write("1\n")
        p.stdin.write("1\n")
        p.stdin.write('dsc.txt\n')
        p.stdin.write("1\n")
        p.stdin.write("1\n")
        p.stdin.write("2\n")
        p.stdin.write("1\n")
        p.stdin.write("\n")
        p.wait()
        input.close()
        output.close()

        while not os.path.exists('in.txt'):
            time.sleep(1)
        p = subprocess.Popen('./guide < in.txt', shell=True)
        p.wait()

        output = file('out.txt', 'r')
        lines = output.readlines()
        output.close()

        start_index, end_index, counter = 0, 0, 0
        for line in lines:
            if line == ' Classification tree:\n':
                start_index = counter+2
            if line == ' ***************************************************************\n':
                end_index = counter-1
            counter += 1
        tree = self.decision_tree_from_text(lines[start_index:end_index])

        self.remove_files()

        return tree

    def decision_tree_from_text(self, lines):

        dt = DecisionTree()

        if '<=' in lines[0] or '>' in lines[0]:
            # Intermediate node
            node_name = lines[0].split(':')[0].lstrip()
            label, value = lines[0].split(':')[1].split('<=')
            label = label.lstrip().rstrip()
            value = value.lstrip().split()[0]
            dt.label = label
            dt.value = value
            dt.left = self.decision_tree_from_text(lines[1:])
            counter = 1
            while lines[counter].split(':')[0].lstrip() != node_name: counter+=1
            dt.right = self.decision_tree_from_text(lines[counter+1:])
        else:
            # Terminal node
            dt.label = lines[0].split(':')[1].lstrip()

        return dt

    def create_desc_and_data_file(self, training_feature_vectors, labels):
        dsc = open("dsc.txt", "w")
        data = open("data.txt", "w")

        dsc.write("data.txt\n")
        dsc.write("\"?\"\n")
        dsc.write("1\n")
        count = 1
        for col in training_feature_vectors.columns:
            dsc.write(str(count) + ' ' + str(col) + ' n\n')
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
        # os.remove('out.txt')


datasets = load_all_datasets()
guide = GUIDEConstructor()
for dataset in datasets:
    df = dataset['dataframe']
    label_col = dataset['label_col']
    feature_cols = dataset['feature_cols']
    X = df[feature_cols]
    y = df[label_col]
    guide.construct_tree(X, y)
