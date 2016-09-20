"""
    Written by Kiani Lannoye & Gilles Vandewiele
    Commissioned by UGent.

    Design of a diagnose- and follow-up platform for patients with chronic headaches
"""

import Orange
from pandas import DataFrame

from constructors.treeconstructor import TreeConstructor
import decisiontree
from pandas_to_orange import df2table


class C45Constructor(TreeConstructor):
    """
    This class contains an implementation of C4.5, written by Quinlan. It uses an extern library
    for this called Orange.
    """

    def __init__(self, gain_ratio=False, cf=0.15):
        self.gain_ratio = gain_ratio
        self.cf = cf

    def get_name(self):
        return "C4.5"

    def construct_tree(self, training_feature_vectors, labels):
        # First call df2table on the feature table
        orange_feature_table = df2table(training_feature_vectors)

        # Convert classes to strings and call df2table
        orange_labels_table = df2table(DataFrame(labels.map(str)))

        # Merge two tables
        orange_table = Orange.data.Table([orange_feature_table, orange_labels_table])

        return self.orange_dt_to_my_dt(Orange.classification.tree.C45Learner(orange_table, gain_ratio=self.gain_ratio,
                                                                             cf=self.cf, min_objs=2, subset=False).tree)

    def orange_dt_to_my_dt(self, orange_dt_root):
        # Check if leaf
        if orange_dt_root.node_type == Orange.classification.tree.C45Node.Leaf:
            return decisiontree.DecisionTree(left=None, right=None, label=str(int(orange_dt_root.leaf)), data=None, value=None)
        else:
            dt = decisiontree.DecisionTree(label=orange_dt_root.tested.name, data=None, value=orange_dt_root.cut)
            dt.left = self.orange_dt_to_my_dt(orange_dt_root.branch[0])
            dt.right = self.orange_dt_to_my_dt(orange_dt_root.branch[1])
            return dt

    #
    # def __init__(self, gain_ratio=False, cf=0.15):
    #     self.gain_ratio = gain_ratio
    #     self.cf = cf
    #
    # def get_name(self):
    #     return "C4.5"
    #
    # def construct_tree(self, data):
    #     # First call df2table on the feature table
    #     orange_table = df2table(data)
    #
    #     print(orange_table)
    #
    #     return self.orange_dt_to_my_dt(Orange.classification.tree.TreeLearner(orange_table))
    #     # return self.orange_dt_to_my_dt(Orange.classification.tree.C45Learner(orange_table, gain_ratio=self.gain_ratio,
    #     #                                                                      cf=self.cf, min_objs=2).tree)
    #
    # def orange_dt_to_my_dt(self, orange_dt_root):
    #     # Check if leaf
    #     print(orange_dt_root.__dict__)
    #     if orange_dt_root.node_type == Orange.classification.tree.C45Node.Leaf:
    #         return DecisionTree(left=None, right=None, label=str(int(orange_dt_root.leaf)+1), data=None, value=None)
    #     else:
    #         dt = DecisionTree(label=orange_dt_root.tested.name, data=None, value=orange_dt_root.cut)
    #         dt.left = self.orange_dt_to_my_dt(orange_dt_root.branch[0])
    #         dt.right = self.orange_dt_to_my_dt(orange_dt_root.branch[1])
    #         return dt
