"""
    Written by Kiani Lannoye & Gilles Vandewiele
    Commissioned by UGent.

    Design of a diagnose- and follow-up platform for patients with chronic headaches
"""

import subprocess
from pandas import Series

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from constructors.treeconstructor import TreeConstructor
import decisiontree


class CARTConstructor(TreeConstructor):
    """
    This class contains an implementation of CART, written by Breiman. It uses an extern library
    for this called sklearn.
    """

    def __init__(self, criterion='gini', min_samples_leaf=1, min_samples_split=2, max_depth=10):
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion

    def get_name(self):
        return "CART"

    def cross_validation(self, data, k):
        return KFold(len(data.index), n_folds=k, shuffle=True)

    def construct_tree(self, training_feature_vectors, labels):
        """
        This method constructs an sklearn decision tree and trains it with the training data
        The sklearn decision tree classifier is stored in the CARTconstructor.dt

        :param training_feature_vectors: the feature vectors of the training samples
        :param labels: the labels of the training samples
        :return: void
        """
        self.features = list(training_feature_vectors.columns)
        # print"* features:", self.features

        self.y = labels.values
        self.X = training_feature_vectors[self.features]


        self.dt = DecisionTreeClassifier(criterion=self.criterion, min_samples_leaf=self.min_samples_leaf,
                                         min_samples_split=self.min_samples_leaf, max_depth=self.max_depth)
        self.dt.fit(self.X, self.y)

        return self.convertToTree()

    def calculate_error_rate(self, tree, testing_feature_vectors, labels):
        return 1-tree.dt.score(testing_feature_vectors, labels)

    def post_prune(self, tree, testing_feature_vectors, labels, significance=0.125):
        pass

    def visualize_tree(tree, feature_names, labelnames, filename):
        """Create tree png using graphviz.

        Args
        ----
        tree -- scikit-learn DecsisionTree.
        feature_names -- list of feature names.
        """
        labels = Series(labelnames.values.ravel()).unique()
        labels.sort()
        labels = map(str, labels)
        # labels = labelnames.unique()
        # print labels
        with open(filename + ".dot", 'w') as f:
            export_graphviz(tree.dt, out_file=f,
                            feature_names=feature_names, class_names=labels)

        command = ["dot", "-Tpdf", filename + ".dot", "-o", filename + ".pdf"]
        try:
            subprocess.check_call(command)
        except:
            exit("Could not run dot, ie graphviz, to "
                 "produce visualization")

    def convertToTree(self, verbose=False):
        # Using those arrays, we can parse the tree structure:
            # label = naam feature waarop je splitst
            # value = is de value van de feature waarop je splitst
            # ownDecisionTree.


        n_nodes = self.dt.tree_.node_count
        children_left = self.dt.tree_.children_left
        children_right = self.dt.tree_.children_right
        feature = self.dt.tree_.feature
        threshold = self.dt.tree_.threshold
        classes = self.dt.classes_

        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes)
        decision_trees = [None] * n_nodes
        for i in range(n_nodes):
            decision_trees[i] = decisiontree.DecisionTree()
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
        if verbose:
            print("The binary tree structure has %s nodes and has "
              "the following tree structure:"
              % n_nodes)

        for i in range(n_nodes):

            if children_left[i] > 0:
                decision_trees[i].left = decision_trees[children_left[i]]

            if children_right[i] > 0:
                decision_trees[i].right = decision_trees[children_right[i]]

            if is_leaves[i]:
                # decision_trees[i].label = self.dt.classes_[self.dt.tree_.value[i][0][1]]
                decision_trees[i].label = self.dt.classes_[np.argmax(self.dt.tree_.value[i][0])]
                decision_trees[i].value = None
                # if verbose:
                #     print(bcolors.OKBLUE + "%snode=%s leaf node." % (node_depth[i] * "\t", i)) + bcolors.ENDC
            else:
                decision_trees[i].label = self.features[feature[i]]
                decision_trees[i].value = threshold[i]

                # if verbose:
                #     print("%snode=%s test node: go to node %s if %s %s <= %s %s else to "
                #       "node %s."
                #       % (node_depth[i] * "\t",
                #          i,
                #          children_left[i],
                #          bcolors.BOLD,
                #          self.features[feature[i]],
                #          threshold[i],
                #          bcolors.ENDC,
                #          children_right[i],
                #          ))
        return decision_trees[0]