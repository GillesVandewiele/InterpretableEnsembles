"""
    Written by Kiani Lannoye & Gilles Vandewiele
    Commissioned by UGent.

    Design of a diagnose- and follow-up platform for patients with chronic headaches
"""
from copy import deepcopy

import sklearn
from graphviz import Source
import matplotlib.pyplot as plt
import numpy as np
import json
import operator

from sklearn.cross_validation import StratifiedKFold


class DecisionTree(object):
    """
    This class contains the main object used throughout our project: a decision tree. It contains methods
    to visualise and evaluate the trees.
    """
    def __init__(self, right=None, left=None, label='', value=None, data=None, parent=None):
        """
        Create a node of the decision tree
        :param right: right child, followed when a feature_value > value
        :param left: left child, followed when feature_value <= value
        :param label: string representation of the attribute the node splits on
                      (feature vector must be dict with same strings and values)
        :param value: the value where the node splits on (if None, then we're in a leaf)
        :param data: list of samples in the subtree
        """
        self.right = right
        self.left = left
        self.label = label
        self.value = value
        self.data = data
        self.parent = parent
        self.class_probabilities = {}

    def visualise(self, output_path, _view=True, with_pruning_ratio=False, show_probabilities=True):
        """
        visualise the tree, calling convert_node_to_dot
        :param output_path: where the file needs to be saved
        :param with_pruning_ratio: if true, the error rate will be printed too
        :param show_probabilities: if this is True, probabilities will be plotted in the leafs too
        :param _view: open the pdf after generation or not
        """
        src = Source(self.convert_to_dot(_with_pruning_ratio=with_pruning_ratio, show_probabilities=show_probabilities))
        src.render(output_path, view=_view)

    def get_number_of_subnodes(self, count=0):
        """
        Private method using in convert_node_to_dot, in order to give the right child of a node the right count
        :param count: intern parameter, don't set it
        :return: the number of subnodes of a specific node, not including himself
        """
        if self.value is None:
            return count
        else:
            return self.left.get_number_of_subnodes(count=count + 1) + self.right.get_number_of_subnodes(
                count=count + 1)

    def convert_node_to_dot(self, count=1, _with_pruning_ratio=False, show_probabilities=True):
        """
        Convert node to dot format in order to visualize our tree using graphviz
        :param count: parameter used to give nodes unique names
        :param _with_pruning_ratio: if true, the error rate will be printed too
        :param show_probabilities: if this is True, probabilities will be plotted in the leafs too
        :return: intermediate string of the tree in dot format, without preamble (this is no correct dot format yet!)
        """
        ratio_string = ('(' + str(self.pruning_ratio) + ')') if _with_pruning_ratio else ''
        if self.value is None:
            if len(self.class_probabilities) > 0 and show_probabilities:
                s = 'Node' + str(count) + ' [label="' + str(self.label) + ratio_string + '\n'+self.class_probabilities.__str__()+'" shape="box"];\n'
            else:
                s = 'Node' + str(count) + ' [label="' + str(self.label) + '" shape="box"];\n'
        else:
            if len(self.class_probabilities) > 0 and show_probabilities:
                s = 'Node' + str(count) + ' [label="' + str(self.label) + ' <= ' + str(self.value) + ratio_string + '\n'+self.class_probabilities.__str__() +'"];\n'
            else:
                s = 'Node' + str(count) + ' [label="' + str(self.label) + ' <= ' + str(self.value) + ratio_string + '"];\n'
            s += self.left.convert_node_to_dot(count=count + 1, _with_pruning_ratio=_with_pruning_ratio)
            s += 'Node' + str(count) + ' -> ' + 'Node' + str(count + 1) + ' [label="true"];\n'
            number_of_subnodes = self.left.get_number_of_subnodes()
            s += self.right.convert_node_to_dot(count=count + number_of_subnodes + 2,
                                                _with_pruning_ratio=_with_pruning_ratio)
            s += 'Node' + str(count) + '->' + 'Node' + str(count + number_of_subnodes + 2) + ' [label="false"];\n'

        return s

    def convert_to_dot(self, _with_pruning_ratio=False, show_probabilities=True):
        """
        Wrapper around convert_node_to_dot (need some preamble and close with })
        :param _with_pruning_ratio: if true, the error rate will be printed too
        :param show_probabilities: if this is True, probabilities will be plotted in the leafs too
        :return: the tree in correct dot format
        """
        s = 'digraph DT{\n'
        s += 'node[fontname="Arial"];\n'
        s += self.convert_node_to_dot(_with_pruning_ratio=_with_pruning_ratio, show_probabilities=show_probabilities)
        s += '}'
        return s

    def to_string(self, tab=0):
        if self.value is None:
            print('\t' * tab + '[', self.label, ']')
        else:
            print('\t' * tab + self.label, ' <= ', str(self.value))
            print('\t' * (tab + 1) + 'LEFT:')
            self.left.to_string(tab=tab + 1)
            print('\t' * (tab + 1) + 'RIGHT:')
            self.right.to_string(tab=tab + 1)

    def evaluate(self, feature_vector):
        """
        Recursive method to evaluate a feature_vector, the feature_vector must be a dict, having the same
        string representations of the attributes as the representations in the tree
        :param feature_vector: the feature_vector to evaluate
        :return: a class label
        """
        if self.value is None:
            return self.label
        else:
            # feature_vector should only contain 1 row
            if feature_vector[self.label] <= self.value:
                return self.left.evaluate(feature_vector)
            else:
                return self.right.evaluate(feature_vector)

    """
        {
        "name" : "0", "rule" : "null",
        "children" : [{ "name" : "2", "rule" : "sunny",
                        "children" : [{ "name" : "no(3/100%)", "rule" : "high" },
                                      { "name" : "yes(2/100%)", "rule" : "normal" }]},
                                      { "name" : "yes(4/100%)", "rule" : "overcast" },
                                      { "name" : "3", "rule" : "rainy",
                                        "children" : [{ "name" : "no(2/100%)", "rule" : "TRUE" },
                                                      { "name" : "yes(3/100%)", "rule" : "FALSE" }
                                                     ]
                                      }
                                     ]
                        }
                     ]
        }
    """
    def to_json(self):
        json = "{\n"
        json += "\t\"name\": \"" + str(self.label) + " <= " + str(self.value) + "\",\n"
        json += "\t\"rule\": \"null\",\n"
        json += "\t\"children\": [\n"
        json += DecisionTree.node_to_json(self.left, "True")+",\n"
        json += DecisionTree.node_to_json(self.right, "False")+"\n"
        json += "\t]\n"
        json += "}\n"
        return json

    @staticmethod
    def node_to_json(node, rule, count=2):
        json = "\t"*count + "{\n"
        if node.value is None:
            if len(node.class_probabilities) > 0:
                json += "\t"*count + "\"name\": \"" + str(node.label) + "( " + str(node.class_probabilities) + ")\",\n"
            else:
                json += "\t"*count + "\"name\": \"" + str(node.label) + " \",\n"
            json += "\t"*count + "\"rule\": \"" + rule + "\"\n"
        else:
            json += "\t"*count + "\"name\": \"" + str(node.label) + " <= " + str(node.value) + "\",\n"
            json += "\t"*count + "\"rule\": \"" + rule + "\",\n"
            json += "\t"*count + "\"children\": [\n"
            json += DecisionTree.node_to_json(node.left, "True", count=count+1)+",\n"
            json += DecisionTree.node_to_json(node.right, "False", count=count+1)+"\n"
            json += "\t"*count + "]\n"
        json += "\t"*count + "}"

        return json

    @staticmethod
    def from_json(json_file):
        tree_json = json.loads(json_file)
        tree = DecisionTree()
        split_name = tree_json['name'].split(" <= ")
        label, value = split_name[0], split_name[1]
        tree.label = label
        tree.value = value
        tree.left = DecisionTree.json_to_node(tree_json['children'][0])
        tree.right = DecisionTree.json_to_node(tree_json['children'][1])
        return tree

    @staticmethod
    def json_to_node(_dict):
        tree = DecisionTree()
        split_name = _dict['name'].split(" <= ")
        if len(split_name) > 1:
            label, value = split_name[0], split_name[1]
            tree.label = label
            tree.value = value
            if 'children' in _dict:
                tree.left = DecisionTree.json_to_node(_dict['children'][0])
                tree.right = DecisionTree.json_to_node(_dict['children'][1])
        else:
            tree.label = split_name[0]
            tree.value = None
            tree.left = None
            tree.right = None
        return tree

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

    @staticmethod
    def plot_confusion_matrix(actual_labels, predicted_labels, normalized=False, plot=False):
        confusion_matrix = sklearn.metrics.confusion_matrix(actual_labels, predicted_labels)
        if plot:
            confusion_matrix.plot(normalized=normalized)
            plt.show()

        return confusion_matrix

    def calc_probabilities(self):
        if self.value is not None:
            self.left.calc_probabilities()
            self.right.calc_probabilities()
        else:
            sum_values = sum(self.class_probabilities.itervalues())
            if sum_values != 0:
                factor = 1.0/sum_values
            else:
                factor = 0
            for i in self.class_probabilities:
                self.class_probabilities[i] = round(self.class_probabilities[i]*factor, 2)
            return

    @staticmethod
    def init_tree(tree, labels):
        for label in np.unique(labels):
            tree.class_probabilities[str(label)] = 0.0

        if tree.value is not None:
            DecisionTree.init_tree(tree.left, labels)
            DecisionTree.init_tree(tree.right, labels)

    def populate_samples(self, feature_vectors, labels):
        index = 0
        DecisionTree.init_tree(self, np.unique(labels))
        for _index, feature_vector in feature_vectors.iterrows():
            current_node = self
            while current_node.value is not None:
                current_node.class_probabilities[str(labels[index])] += 1
                if feature_vector[current_node.label] <= current_node.value:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            current_node.class_probabilities[str(labels[index])] += 1
            index += 1

    ############################### PRUNING FROM HERE ON #####################################

    def set_parents(self):
        if self.value is not None:
            self.left.parent = self
            self.left.set_parents()
            self.right.parent = self
            self.right.set_parents()

    def count_nodes(self):
        if self.value is None:
            return 1
        else:
            return self.left.count_nodes() + self.right.count_nodes() + 1

    def count_leaves(self):
        if self.value is None:
            return 1
        else:
            return self.left.count_leaves() + self.right.count_leaves()

    def get_leaves(self):
        if self.value is None:
            return [self]
        else:
            leaves=[]
            leaves.extend(self.left.get_leaves())
            leaves.extend(self.right.get_leaves())
            return leaves

    def get_nodes(self):
        if self.value is not None:
            nodes = [self]
        else:
            nodes = []
        if self.left is not None:
            nodes.extend(self.left.get_nodes())
        if self.right is not None:
            nodes.extend(self.right.get_nodes())
        return nodes


    def calc_leaf_error(self, total_train_samples):
        # for leaf in self.get_leaves():
        #     print leaf.class_probabilities
        #     print sum(leaf.class_probabilities.values())/total_train_samples, 1 - leaf.class_probabilities[str(leaf.label)]/sum(leaf.class_probabilities.values())
        #     print (sum(leaf.class_probabilities.values())/total_train_samples) *  \
        #             (1 - leaf.class_probabilities[str(leaf.label)]/sum(leaf.class_probabilities.values()))
        return sum([(sum(leaf.class_probabilities.values()) / total_train_samples) *
                    (1 - leaf.class_probabilities[str(leaf.label)]/sum(leaf.class_probabilities.values()))
                    for leaf in self.get_leaves()])

    def calc_node_error(self, total_train_samples):
        return (1 - max(self.class_probabilities.iteritems(), key=operator.itemgetter(1))[1]/sum(self.class_probabilities.values())) \
               * (sum(self.class_probabilities.values()) / total_train_samples)

    def calculate_alpha(self, total_train_samples):
        return (self.calc_node_error(total_train_samples) - self.calc_leaf_error(total_train_samples)) / (self.count_leaves() - 1)

    def calculate_cost_complexity(self, total_train_samples, alpha):
        return self.calc_leaf_error(total_train_samples) + alpha * self.count_leaves()

    def prune_node(self, node):
        if self == node:
            self.label = max(self.class_probabilities.items(), key=operator.itemgetter(1))[0]
            self.value = None
            self.right = None
            self.left = None
        else:
            if self.left is not None and self.left.value is not None:
                self.left.prune_node(node)
            if self.right is not None and self.right.value is not None:
                self.right.prune_node(node)

    def generate_subtree(self, total_train_samples, alphas={}):
        # print self.label, self.value
        if self.value is not None:
            calc_alpha = self.calculate_alpha(total_train_samples)
            alphas[self] = (calc_alpha, self.count_nodes())
            if self.left.value is not None:
                self.left.generate_subtree(total_train_samples, alphas)
            if self.right.value is not None:
                self.right.generate_subtree(total_train_samples, alphas)
        return alphas

    def generate_subtree_sequence(self, total_train_samples):
        subtrees = {}
        current_tree = deepcopy(self)
        while current_tree.left is not None or current_tree.right is not None:
            generated_trees = current_tree.generate_subtree(total_train_samples, {})
            print generated_trees.values()
            best = min(generated_trees.items(), key=operator.itemgetter(1))
            tree, alpha = best[0], best[1][0]
            current_tree.prune_node(tree)
            subtrees[deepcopy(current_tree)] = alpha
        return subtrees

    def minimize_cost_complexity(self, total_train_samples, alpha):
        while 1:
            min_complexity, min_nodes = self.calculate_cost_complexity(total_train_samples, alpha), self.count_nodes()
            best_node_to_prune = None
            print [(node.label, node.value) for node in self.get_nodes()]
            tree = deepcopy(self)
            for node in tree.get_nodes():
                print node.label, node.value
                print 'nodes before pruning:', tree.count_nodes()
                tree.prune_node(node)
                print 'nodes after pruning:', tree.count_nodes()
                complexity, nodes = tree.calculate_cost_complexity(total_train_samples, alpha), tree.count_nodes()
                print '----->', complexity, nodes
                if (complexity, nodes) <= (min_complexity, min_nodes):
                    best_node_to_prune = node
                    min_complexity = complexity
                    min_nodes = nodes

            if best_node_to_prune is not None:
                print 'best_node:', best_node_to_prune.label, best_node_to_prune.value
                self.prune_node(best_node_to_prune)
            else:
                break

    def cost_complexity_pruning(self, feature_vectors, labels, tree_constructor, n_folds=3):
        # TODO: implement pruning (ftp://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/TREE-pruning.pdf) or (http://mlwiki.org/index.php/Cost-Complexity_Pruning)
        self.set_parents()
        self.populate_samples(feature_vectors, labels.values)
        root_samples = sum(self.class_probabilities.values())

        subtrees = self.generate_subtree_sequence(root_samples)
        print subtrees

        skf = StratifiedKFold(labels, n_folds=n_folds, shuffle=True)
        for train_index, test_index in skf:
            X_train = feature_vectors.iloc[train_index, :]
            y_train = labels.iloc[train_index]
            X_test = feature_vectors.iloc[test_index, :]
            y_test = labels.iloc[test_index]
            constructed_tree = tree_constructor.construct_tree(X_train, y_train)
            constructed_tree.populate_samples(X_train, y_train.values)
            # constructed_tree.visualise('test')
            constructed_tree.minimize_cost_complexity(root_samples, 0.05)
            print constructed_tree.calculate_cost_complexity(root_samples, 0.075)
            constructed_tree.prune_node(constructed_tree.right)
            print constructed_tree.calculate_cost_complexity(root_samples, 0.075)
            raw_input('....')

        pass





        # best = min(self.generate_subtree(root_samples).items(), key=self.generate_subtree(root_samples).get)
        # prune_tree, alpha = best[0], best[1][0]
        # self.prune_node(prune_tree)
        # self.visualise('pruned')
        # print 'Iteration 1'
        # print (self.count_leaves() - 1)
        # print self.calc_node_error(root_samples)
        # print self.calc_leaf_error(root_samples)
        # g_t = (self.calc_node_error(root_samples) - self.calc_leaf_error(root_samples)) / (self.count_leaves() - 1)
        # print g_t
        # print '============='
        # print (self.right.count_leaves() - 1)
        # print self.right.calc_node_error(root_samples)
        # print self.right.calc_leaf_error(root_samples)
        # g_t = (self.right.calc_node_error(root_samples)- self.right.calc_leaf_error(root_samples)) / (self.right.count_leaves() - 1)
        # print g_t
        # print '============='
        # print (self.right.right.count_leaves() - 1)
        # print self.right.right.calc_node_error(root_samples)
        # print self.right.right.calc_leaf_error(root_samples)
        # g_t = (self.right.right.calc_node_error(root_samples)- self.right.right.calc_leaf_error(root_samples)) / (self.right.right.count_leaves() - 1)
        # print g_t
        #
        #
        # print 'Iteration 2'
        # self.right.right.label = 0
        # self.right.right.value = None
        # self.right.right.left = None
        # self.right.right.right = None
        # self.visualise('prune_1')
        # print (self.count_leaves() - 1)
        # print self.calc_node_error(root_samples)
        # print self.calc_leaf_error(root_samples)
        # g_t = (self.calc_node_error(root_samples) - self.calc_leaf_error(root_samples)) / (self.count_leaves() - 1)
        # print g_t
        # print '============='
        # print (self.right.count_leaves() - 1)
        # print self.right.calc_node_error(root_samples)
        # print self.right.calc_leaf_error(root_samples)
        # g_t = (self.right.calc_node_error(root_samples)- self.right.calc_leaf_error(root_samples)) / (self.right.count_leaves() - 1)
        # print g_t




        # g_t = (max(self.right.class_probabilities.iteritems(), key=operator.itemgetter(1))[1]/sum(self.right.class_probabilities.values())
        #        - self.calc_leaf_error(root_samples)) / (self.count_leaves() - 1)
        # print self.calc_leaf_error()
