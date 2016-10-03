# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys

from ISM_v3 import bootstrap
from constructors.c45orangeconstructor import C45Constructor
from constructors.cartconstructor import CARTConstructor
# from data.load_datasets import load_heart

sys.path.append('../')
from defragTrees import *

import numpy as np

from rpy2.robjects.packages import importr
import pandas.rpy.common as com
import rpy2.robjects as ro
import re


class Rule:

    def __init__(self, feature, test, value):
        self.feature = feature
        self.test = test
        self.value = value

    def evaluate(self, feature_vector):
        if self.value is None:
            return True
        elif self.test == '==':
            return feature_vector[self.feature] == self.value
        elif self.test == '>':
            return feature_vector[self.feature] > self.value
        else:
            return feature_vector[self.feature] <= self.value


class RuleSet:

    def __init__(self, index, rules, prediction):
        self.index = index
        self.rules = rules
        self.prediction = prediction

    def evaluate(self, feature_vector):
        for rule in self.rules:
            if not rule.evaluate(feature_vector): return False, -1
        return True, self.prediction


class OrderedRuleList:

    def __init__(self, rule_list):
        self.rule_list = rule_list

    def evaluate(self, feature_vector):
        for ruleset in sorted(self.rule_list, key=lambda x: x.index):  # Sort to make sure they are evaluated in order
            rule_evaluation_result, rule_evaluation_pred = ruleset.evaluate(feature_vector)
            if rule_evaluation_result: return rule_evaluation_pred
        return None

    def evaluate_multiple(self, feature_vectors):
        """
        Wrapper method to evaluate multiple vectors at once (just a for loop where evaluate is called)
        :param feature_vectors: the feature_vectors you want to evaluate
        :return: list of class labels
        """
        results = []

        for _index, feature_vector in feature_vectors.iterrows():
            results.append(self.evaluate(feature_vector))

        return np.asarray(results)


class inTreesClassifier:

    def __init__(self):
        pass

    def convert_to_r_dataframe(self, df, strings_as_factors=False):
        """
        Convert a pandas DataFrame to a R data.frame.

        Parameters
        ----------
        df: The DataFrame being converted
        strings_as_factors: Whether to turn strings into R factors (default: False)

        Returns
        -------
        A R data.frame

        """

        import rpy2.rlike.container as rlc

        columns = rlc.OrdDict()

        # FIXME: This doesn't handle MultiIndex

        for column in df:
            value = df[column]
            value_type = value.dtype.type

            if value_type == np.datetime64:
                value = com.convert_to_r_posixct(value)
            else:
                value = [item if pd.notnull(item) else com.NA_TYPES[value_type]
                         for item in value]

                value = com.VECTOR_TYPES[value_type](value)

                if not strings_as_factors:
                    I = ro.baseenv.get("I")
                    value = I(value)

            columns[column] = value

        r_dataframe = ro.DataFrame(columns)
        del columns

        r_dataframe.rownames = ro.StrVector(list(df.index))
        r_dataframe.colnames = list(df.columns)

        return r_dataframe

    def tree_to_R_object(self, tree, feature_mapping):
        node_mapping = {}
        nodes = tree.get_nodes()
        nodes.extend(tree.get_leaves())
        for i, node in enumerate(nodes):
            node_mapping[node] = i+1
        vectors = []
        for node in nodes:
            if node.value is not None:
                vectors.append([node_mapping[node], node_mapping[node.left], node_mapping[node.right],
                                feature_mapping[node.label], node.value, 1, 0])
            else:
                vectors.append([node_mapping[node], 0, 0, 0, 0.0, -1, node.label])

        df = pd.DataFrame(vectors)
        df.columns = ['id', 'left daughter', 'right daughter', 'split var', 'split point', 'status', 'prediction']
        df = df.set_index('id')
        df.index.name = None

        return self.convert_to_r_dataframe(df)

    def construct_rule_list(self, train_df, label_col, tree_constructors, nr_bootstraps=3):
        y_train = train_df[label_col]
        X_train = train_df.copy()
        X_train = X_train.drop(label_col, axis=1)

        importr('randomForest')
        importr('inTrees')

        ro.globalenv["X"] = com.convert_to_r_dataframe(X_train)
        ro.globalenv["target"] = ro.FactorVector(y_train.values.tolist())

        feature_mapping = {}
        feature_mapping_reverse = {}
        for i, feature in enumerate(X_train.columns):
            feature_mapping[feature] = i + 1
            feature_mapping_reverse[i + 1] = feature

        treeList = []
        for tree in bootstrap(train_df, label_col, tree_constructors, nr_classifiers=nr_bootstraps):
            if tree.count_nodes() > 1: treeList.append(self.tree_to_R_object(tree, feature_mapping))

        ro.globalenv["treeList"] = ro.Vector([len(treeList), ro.Vector(treeList)])
        ro.r('names(treeList) <- c("ntree", "list")')

        rules = ro.r('buildLearner(getRuleMetric(extractRules(treeList, X), X, target), X, target)')
        rules=list(rules)
        conditions=rules[int(0.6*len(rules)):int(0.8*len(rules))]
        predictions=rules[int(0.8*len(rules)):]

        # Create a OrderedRuleList
        rulesets = []
        for idx, (condition, prediction) in enumerate(zip(conditions, predictions)):
            # Split each condition in Rules to form a RuleSet
            rulelist = []
            condition_split = [x.lstrip().rstrip() for x in condition.split('&')]
            for rule in condition_split:
                feature = feature_mapping_reverse[int(re.findall(r',[0-9]+]', rule)[0][1:-1])]

                lte = re.findall(r'<=', rule)
                gt = re.findall(r'>', rule)
                eq = re.findall(r'==', rule)
                cond = lte[0] if len(lte) else (gt[0] if len(gt) else eq[0])

                extract_value = re.findall(r'[=>]-?[0-9\.]+', rule)
                if len(extract_value):
                    value = float(re.findall(r'[=>]-?[0-9\.]+', rule)[0][1:])
                else:
                    feature = 'True'
                    value = None

                rulelist.append(Rule(feature, cond, value))
            rulesets.append(RuleSet(idx, rulelist, prediction))

        return OrderedRuleList(rulesets)


# heart, features, label_col, dataset_name = load_heart()
# heart = heart.iloc[np.random.permutation(len(heart))].reset_index(drop=True)
# heart_train = heart.head(int(0.75*len(heart)))
# heart_test = heart.tail(int(0.25*len(heart)))
#
# X_train = heart_train.drop(label_col, axis=1)
# y_train = heart_train[label_col]
# X_test = heart_test.drop(label_col, axis=1)
# y_test = heart_test[label_col]
#
# c45 = C45Constructor()
# cart = CARTConstructor()
# inTrees = inTreesClassifier()
#
# orl = inTrees.construct_rule_list(heart_train, label_col, [c45, cart], nr_bootstraps=3)
# print orl.evaluate_multiple(heart_test)
