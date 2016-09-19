"""
    Written by Kiani Lannoye & Gilles Vandewiele
    Commissioned by UGent.

    Design of a diagnose- and follow-up platform for patients with chronic headaches
"""

import Orange
from Orange.classification.rules import CN2UnorderedLearner
from Orange.classification.rules import CN2SDUnorderedLearner
from pandas import DataFrame
import numpy as np

from pandas_to_orange import df2table


class CN2UnorderedConstructor:
    """
    This class contains an implementation of CN2. It uses an extern library for this called Orange.
    """

    def __init__(self, beam_width=5, alpha=1.0):
        self.model = None
        self.feature_domain = None
        self.beam_width = beam_width
        self.alpha = alpha

    def get_name(self):
        return "CN2"

    def rules_to_decision_space(self):

        def selectSign(oper):
            if oper == Orange.data.filter.ValueFilter.Less:
                return "<"
            elif oper == Orange.data.filter.ValueFilter.LessEqual:
                return "<="
            elif oper == Orange.data.filter.ValueFilter.Greater:
                return ">"
            elif oper == Orange.data.filter.ValueFilter.GreaterEqual:
                return ">="
            else:
                return "="

        for rule in self.model.rules:
            conds = rule.filter.conditions
            domain = rule.filter.domain
            print ' RULE: IF '
            for cond in conds:
                print 'Name:', domain[cond.position].name, 'Sign:', selectSign(cond.oper) ,'Value:', str(cond.ref)
            distr = [value for value in rule.class_distribution]
            print ' THEN ', str(np.divide(distr, sum(distr)))


    def extract_rules(self, training_feature_vectors, labels):
        # First call df2table on the feature table
        orange_feature_table = df2table(training_feature_vectors)
        self.feature_domain = orange_feature_table.domain

        # Convert classes to strings and call df2table
        orange_labels_table = df2table(DataFrame(labels.map(str)))

        # Merge two tables
        orange_table = Orange.data.Table([orange_feature_table, orange_labels_table])

        self.model = CN2UnorderedLearner(orange_table, beam_width=self.beam_width, alpha=self.alpha)
        return self.model

    def print_rules(self,):
        for rule in self.model.rules:
            print Orange.classification.rules.rule_to_string(rule)

    def classify(self, vectors):
        predictions = []
        for i in range(len(vectors)):
            inst = Orange.data.Instance(self.feature_domain, [vectors.iloc[i, :][column.name] for column in self.feature_domain.variables])
            predictions.append(self.model(inst, result_type=Orange.classification.Classifier.GetBoth, ret_rules=True))
        return predictions


#TODO: Check which parameter this creature has and incorporate them



# import pandas as pd
# import numpy as np
# columns = ['ID', 'ClumpThickness', 'CellSizeUniform', 'CellShapeUniform', 'MargAdhesion', 'EpithCellSize', 'BareNuclei',
#            'BlandChromatin', 'NormalNuclei', 'Mitoses', 'Class']
# features = ['ClumpThickness', 'CellSizeUniform', 'CellShapeUniform', 'MargAdhesion', 'EpithCellSize', 'BareNuclei',
#            'BlandChromatin', 'NormalNuclei', 'Mitoses']
# df = pd.read_csv('../data/breast-cancer-wisconsin.data')
# df.columns = columns
# df['Class'] = np.subtract(np.divide(df['Class'], 2), 1)
# df = df.drop('ID', axis=1).reset_index(drop=True)
# df['BareNuclei'] = df['BareNuclei'].replace('?', int(np.mean(df['BareNuclei'][df['BareNuclei'] != '?'].map(int))))
# df = df.applymap(int)
#
# X = df.drop('Class', axis=1).reset_index(drop=True)
# y = df['Class'].reset_index(drop=True)
#
#
# cn2 = CN2UnorderedConstructor()
# cn2.extract_rules(X, y)
# # print X.iloc[4,:]['CellSizeUniform']
# print cn2.classify(X.iloc[4:5,])

