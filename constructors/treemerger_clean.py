import copy
import multiprocessing
import random

from imblearn.over_sampling import SMOTE
from pandas import DataFrame, concat, Series

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import operator

import time
import sys

import sklearn
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import ISM_v3

from decisiontree import DecisionTree
from featuredescriptors import CONTINUOUS

from multiprocessing import Pool


class LineSegment(object):
    """
        Auxiliary class, used for the intersection algorithm
    """
    def __init__(self, lower_bound, upper_bound, region_index):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.region_index = region_index


class DecisionTreeMergerClean(object):

    def decision_tree_to_decision_table(self, tree, feature_vectors):
        """
        Convert each path from the root to a leaf into a region, store it into a table
        :param tree: the constructed tree
        :param feature_vectors: the feature vectors of all samples
        :return: a set of regions in a k-dimensional space (k=|feature_vector|), corresponding to the decision tree
        """
        # Initialize an empty region (will be passed on recursively)
        region = {}
        for column in feature_vectors.columns:
            region[column] = [float("-inf"), float("inf")]
            region["class"] = None
        regions = self.tree_to_decision_table(tree, region, [])
        return regions

    def tree_to_decision_table(self, tree, region, regions):
        """
        Recursive method used to convert the decision tree to a decision_table (do not call this one!)
        """
        left_region = copy.deepcopy(region)  # Take a deepcopy or we're fucked
        right_region = copy.deepcopy(region)  # Take a deepcopy or we're fucked
        left_region[tree.label][1] = tree.value
        right_region[tree.label][0] = tree.value

        if tree.left.value is None:
            if tree.left.class_probabilities is not None:
                left_region["class"] = tree.left.class_probabilities
            else:
                left_region["class"] = {tree.left.label: 1.0}
            regions.append(left_region)
        else:
            self.tree_to_decision_table(tree.left, left_region, regions)

        if tree.right.value is None:
            if tree.right.class_probabilities is not None:
                right_region["class"] = tree.right.class_probabilities
            else:
                left_region["class"] = {tree.left.label: 1.0}
            regions.append(right_region)
        else:
            self.tree_to_decision_table(tree.right, right_region, regions)

        return regions

    def find_lines(self, regions, features, feature_mins, feature_maxs):    #TODO: try to optimize this
        if len(regions) <= 0: return {}

        for region in regions:
            for feature in features:
                if region[feature][0] == float("-inf"):
                    region[feature][0] = feature_mins[feature]
                if region[feature][1] == float("inf"):
                    region[feature][1] = feature_maxs[feature]

        lines = {}
        # First convert the region information into dataframes
        columns = []
        for feature in features:
            columns.append(feature+'_lb')
            columns.append(feature+'_ub')
        columns.append('class')
        regions_df = DataFrame(columns=columns)
        for region in regions:
            entry = []
            for feature in features:
                entry.append(region[feature][0])
                entry.append(region[feature][1])
            entry.append(region['class'])
            regions_df.loc[len(regions_df)] = entry

        for feature in features:
            other_features = list(set(features) - set([feature]))
            lb_bool_serie = [True]*len(regions_df)
            ub_bool_serie = [True]*len(regions_df)
            for other_feature in other_features:
                lb_bool_serie &= (regions_df[other_feature+'_lb'] == feature_mins[other_feature]).values
                ub_bool_serie &= (regions_df[other_feature+'_ub'] == feature_maxs[other_feature]).values

            lower_upper_regions = concat([regions_df[lb_bool_serie], regions_df[ub_bool_serie]])
            lines[feature] = []
            for value in np.unique(lower_upper_regions[lower_upper_regions.duplicated(feature+'_lb', False)][feature+'_lb']):
                if feature_mins[feature] != value and feature_maxs[feature] != value:
                    lines[feature].append(value)

        return lines

    def regions_to_tree_improved(self, features_df, labels_df, regions, features, feature_mins, feature_maxs, max_samples=1):

        lines = self.find_lines(regions, features, feature_mins, feature_maxs)
        lines_keys = [key for key in lines.keys() if len(lines[key]) > 0]
        if lines is None or len(lines) <= 0 or len(lines_keys) <= 0:
            return DecisionTree(label=str(np.argmax(np.bincount(labels_df['cat'].values.astype(int)))), value=None, data=features_df)

        random_label = np.random.choice(lines_keys)
        random_value = np.random.choice(lines[random_label])
        data = DataFrame(features_df)
        data['cat'] = labels_df
        best_split_node = DecisionTree(data=data, label=random_label, value=random_value,
                            left=DecisionTree(data=data[data[random_label] <= random_value]),
                            right=DecisionTree(data=data[data[random_label] > random_value]))
        node = DecisionTree(label=best_split_node.label, value=best_split_node.value, data=best_split_node.data)

        feature_mins_right = feature_mins.copy()
        feature_mins_right[node.label] = node.value
        feature_maxs_left = feature_maxs.copy()
        feature_maxs_left[node.label] = node.value
        regions_left = []
        regions_right = []
        for region in regions:
            if region[best_split_node.label][0] < best_split_node.value:
                regions_left.append(region)
            else:
                regions_right.append(region)
        if len(best_split_node.left.data) >= max_samples and len(best_split_node.right.data) >= max_samples:
            node.left = self.regions_to_tree_improved(best_split_node.left.data.drop('cat', axis=1),
                                                      best_split_node.left.data[['cat']], regions_left, features,
                                                      feature_mins, feature_maxs_left)
            node.right = self.regions_to_tree_improved(best_split_node.right.data.drop('cat', axis=1),
                                                       best_split_node.right.data[['cat']], regions_right, features,
                                                       feature_mins_right, feature_maxs)

        else:
            node.label = str(np.argmax(np.bincount(labels_df['cat'].values.astype(int))))
            node.value = None

        return node

    def intersect(self, line1_lb, line1_ub, line2_lb, line2_ub):
        if line1_ub <= line2_lb: return False
        if line1_lb >= line2_ub: return False
        return True


    def calculate_intersection(self, regions1, regions2, features, feature_maxs, feature_mins):
        """
            Fancy method to calculate intersections. O(d*n*log(n)) instead of O(d*n^2)

            Instead of brute force, we iterate over each possible dimension,
            we project each region to that one dimension, creating a line segment. We then construct a set S_i for each
            dimension containing pairs of line segments that intersect in dimension i. In the end, the intersection
            of all these sets results in the intersecting regions. For all these intersection regions, their intersecting
            region is calculated and added to a new set, which is returned in the end
        :param regions1: first set of regions
        :param regions2: second set of regions
        :param features: list of dimension names
        :return: new set of regions, which are the intersections of the regions in 1 and 2
        """
        print "Merging ", len(regions1), " with ", len(regions2), " regions."
        S_intersections = [None] * len(features)
        for i in range(len(features)):
            # Create B1 and B2: 2 arrays of line segments
            box_set1 = []
            for region_index in range(len(regions1)):
                box_set1.append(LineSegment(regions1[region_index][features[i]][0], regions1[region_index][features[i]][1],
                                            region_index))
            box_set2 = []
            for region_index in range(len(regions2)):
                box_set2.append(LineSegment(regions2[region_index][features[i]][0], regions2[region_index][features[i]][1],
                                            region_index))

            # Sort the two boxsets by their lower bound
            box_set1 = sorted(box_set1, key=lambda segment: segment.lower_bound)
            box_set2 = sorted(box_set2, key=lambda segment: segment.lower_bound)

            # Create a list of unique lower bounds, we iterate over these bounds later
            unique_lower_bounds = []
            for j in range(max(len(box_set1), len(box_set2))):
                if j < len(box_set1) and box_set1[j].lower_bound not in unique_lower_bounds:
                    unique_lower_bounds.append(box_set1[j].lower_bound)

                if j < len(box_set2) and box_set2[j].lower_bound not in unique_lower_bounds:
                    unique_lower_bounds.append(box_set2[j].lower_bound)

            # Sort them
            unique_lower_bounds = sorted(unique_lower_bounds)

            box1_active_set = []
            box2_active_set = []
            intersections = []
            for lower_bound in unique_lower_bounds:
                # Update all active sets, a region is added when it's lower bound is lower than the current one
                # It is removed when its upper bound is higher than the current lower bound
                for j in range(len(box_set1)):
                    if box_set1[j].upper_bound <= lower_bound:
                        if box_set1[j] in box1_active_set:
                            box1_active_set.remove(box_set1[j])
                    elif box_set1[j].lower_bound <= lower_bound:
                        if box_set1[j] not in box1_active_set:
                            box1_active_set.append(box_set1[j])
                    else:
                        break

                for j in range(len(box_set2)):
                    if box_set2[j].upper_bound <= lower_bound:
                        if box_set2[j] in box2_active_set:
                            box2_active_set.remove(box_set2[j])
                    elif box_set2[j].lower_bound <= lower_bound:
                        if box_set2[j] not in box2_active_set:
                            box2_active_set.append(box_set2[j])
                    else:
                        break

                # All regions from the active set of B1 intersect with the regions in the active set of B2
                for segment1 in box1_active_set:
                    for segment2 in box2_active_set:
                        intersections.append((segment1.region_index, segment2.region_index))

            # print "----------------------------->", intersections

            S_intersections[i] = intersections


        # for k in range(len(S_intersections)):
        #     print 'Old', features[k], len(S_intersections[k])

        # The intersection of all these S_i's are the intersecting regions
        intersection_regions_indices = S_intersections[0]
        for k in range(1, len(S_intersections)):
            intersection_regions_indices = self.tuple_list_intersections(intersection_regions_indices, S_intersections[k])


        # print 'Old:', len(intersection_regions_indices)

        # Create a new set of regions
        intersected_regions = []
        for intersection_region_pair in intersection_regions_indices:
            region = {}
            for feature in features:
                region[feature] = [max(regions1[intersection_region_pair[0]][feature][0],
                                       regions2[intersection_region_pair[1]][feature][0]),
                                   min(regions1[intersection_region_pair[0]][feature][1],
                                       regions2[intersection_region_pair[1]][feature][1])]
                # Convert all -inf and inf to the mins and max from those features
                if region[feature][0] == float("-inf"):
                    region[feature][0] = feature_mins[feature]
                if region[feature][1] == float("inf"):
                    region[feature][1] = feature_maxs[feature]
            region['class'] = {}
            for key in set(set(regions1[intersection_region_pair[0]]['class'].iterkeys()) |
                                   set(regions2[intersection_region_pair[1]]['class'].iterkeys())):
                prob_1 = (regions1[intersection_region_pair[0]]['class'][key]
                          if key in regions1[intersection_region_pair[0]]['class'] else 0)
                prob_2 = (regions2[intersection_region_pair[1]]['class'][key]
                          if key in regions2[intersection_region_pair[1]]['class'] else 0)
                if prob_1 and prob_2:
                    region['class'][key] = (regions1[intersection_region_pair[0]]['class'][key] +
                                            regions2[intersection_region_pair[1]]['class'][key]) / 2
                else:
                    if prob_1:
                        region['class'][key] = prob_1
                    else:
                        region['class'][key] = prob_2
            intersected_regions.append(region)

        return intersected_regions

    # def calculate_intersection(self, regions1, regions2, features, feature_maxs, feature_mins):
    #     """
    #         Fancy method to calculate intersections. O(d*n*log(n)) instead of O(d*n^2)
    #
    #         Instead of brute force, we iterate over each possible dimension,
    #         we project each region to that one dimension, creating a line segment. We then construct a set S_i for each
    #         dimension containing pairs of line segments that intersect in dimension i. In the end, the intersection
    #         of all these sets results in the intersecting regions. For all these intersection regions, their intersecting
    #         region is calculated and added to a new set, which is returned in the end
    #     :param regions1: first set of regions
    #     :param regions2: second set of regions
    #     :param features: list of dimension names
    #     :return: new set of regions, which are the intersections of the regions in 1 and 2
    #     """
    #
    #     S_intersections = [None] * len(features)
    #
    #     for i, feature in enumerate(features):    # O(d), d = dimension(data)
    #         # print feature
    #         S_intersections[i] = []
    #         regions1 = sorted(regions1, key=lambda x: x[feature])   # O(d * n * log(n)), n=nr_of_regions
    #         regions2 = sorted(regions2, key=lambda x: x[feature])   # O(d * n * log(n)), n=nr_of_regions
    #
    #         idx1 = 0
    #         for region1 in regions1:
    #             for idx2 in range(idx1, len(regions2)):
    #                 region2 = regions2[idx2]
    #                 if region2[feature][1] < region1[feature][0]:
    #                     idx1 += 1
    #                 if region2[feature][0] >= region1[feature][1]:
    #                     break
    #                 else:
    #                     if self.intersect(region1[feature][0], region1[feature][1], region2[feature][0], region2[feature][1]):
    #                         S_intersections[i].append((region1, region2))
    #
    #     intersecting_regions = S_intersections[0]
    #     for k in range(1, len(S_intersections)):
    #         intersecting_regions = [i for i in intersecting_regions for j in S_intersections[k] if i==j]
    #
    #     # Create a new set of regions
    #     intersected_regions = []
    #     for intersection_region_pair in intersecting_regions:
    #         region = {}
    #         for feature in features:
    #             region[feature] = [max(intersection_region_pair[0][feature][0],
    #                                    intersection_region_pair[1][feature][0]),
    #                                min(intersection_region_pair[0][feature][1],
    #                                    intersection_region_pair[1][feature][1])]
    #             # Convert all -inf and inf to the mins and max from those features
    #             if region[feature][0] == float("-inf"):
    #                 region[feature][0] = feature_mins[feature]
    #             if region[feature][1] == float("inf"):
    #                 region[feature][1] = feature_maxs[feature]
    #         region['class'] = {}
    #         for key in set(set(intersection_region_pair[0]['class'].iterkeys()) |
    #                                set(intersection_region_pair[1]['class'].iterkeys())):
    #             prob_1 = (intersection_region_pair[0]['class'][key]
    #                       if key in intersection_region_pair[0]['class'] else 0)
    #             prob_2 = (intersection_region_pair[1]['class'][key]
    #                       if key in intersection_region_pair[1]['class'] else 0)
    #             if prob_1 and prob_2:
    #                 region['class'][key] = (intersection_region_pair[0]['class'][key] +
    #                                         intersection_region_pair[1]['class'][key]) / 2
    #             else:
    #                 if prob_1:
    #                     region['class'][key] = prob_1
    #                 else:
    #                     region['class'][key] = prob_2
    #         intersected_regions.append(region)
    #
    #     return intersected_regions

    def tuple_list_intersections(self, list1, list2):
        # Make sure the length of list1 is larger than the length of list2
        if len(list2) > len(list1):
            return self.tuple_list_intersections(list2, list1)
        else:
            list1 = set(list1)
            list2 = set(list2)
            intersections = []
            for tuple in list2:
                if tuple in list1:
                    intersections.append(tuple)

            return intersections

    def fitness(self, tree, test_features_df, test_labels_df, cat_name, alpha=1, beta=0):
        return alpha*(1-accuracy_score(test_labels_df[cat_name].values.astype(str),
                                       tree.evaluate_multiple(test_features_df).astype(str))) + beta*tree.count_nodes()

    def mutate_shiftRandom(self, tree, feature_vectors, labels):
        # tree.visualise('beforeMutationShift')
        internal_nodes = list(set(tree.get_nodes()) - set(tree.get_leaves()))
        tree = copy.deepcopy(tree)
        # print 'nr internal nodes =', len(internal_nodes)
        if len(internal_nodes) > 1:
            random_node = np.random.choice(internal_nodes)
            random_value = np.random.choice(np.unique(feature_vectors[random_node.label].values))
            random_node.value = random_value
            tree.populate_samples(feature_vectors, labels)
        # tree.visualise('afterMutationShift')
        # raw_input()
        return tree

    def mutate_swapSubtrees(self, tree, feature_vectors, labels):
        # tree.visualise('beforeMutationSwap')
        tree = copy.deepcopy(tree)
        tree.set_parents()
        nodes = tree.get_nodes()
        if len(nodes) > 1:
            node1 = np.random.choice(nodes)
            node2 = np.random.choice(nodes)
            parent1 = node1.parent
            parent2 = node2.parent

            if parent1 is not None and parent2 is not None: # We don't want the root node
                if parent1.left == node1:
                    parent1.left = node2
                else:
                    parent1.right = node2

                if parent2.left == node2:
                    parent2.left = node1
                else:
                    parent2.right = node1

                tree.populate_samples(feature_vectors, labels)
        # tree.visualise('afterMutationSwap')
        # raw_input()
        return tree

    def tournament_selection_and_merging(self, trees, train_features_df, train_labels_df, test_features_df,
                                         test_labels_df, cat_name, feature_cols, feature_maxs, feature_mins,
                                         max_samples, return_dict, seed, tournament_size=3):
        np.random.seed(seed)
        _tournament_size = min(len(trees) / 2, tournament_size)
        trees = copy.deepcopy(trees)
        best_fitness_1 = sys.float_info.max
        best_tree_1 = None
        for i in range(_tournament_size):
            tree = np.random.choice(trees)
            trees.remove(tree)
            fitness = self.fitness(tree, test_features_df, test_labels_df, cat_name)
            if tree is not None and tree.count_nodes() > 1 and fitness < best_fitness_1:
                best_fitness_1 = fitness
                best_tree_1 = tree
        best_fitness_2 = sys.float_info.max
        best_tree_2 = None
        for i in range(_tournament_size):
            tree = np.random.choice(trees)
            trees.remove(tree)
            fitness = self.fitness(tree, test_features_df, test_labels_df, cat_name)
            if tree is not None and tree.count_nodes() > 1 and fitness < best_fitness_2:
                best_fitness_2 = fitness
                best_tree_2 = tree

        if best_tree_1 is not None and best_tree_2 is not None:
            region1 = self.decision_tree_to_decision_table(best_tree_1, train_features_df)
            region2 = self.decision_tree_to_decision_table(best_tree_2, train_features_df)
            merged_regions = self.calculate_intersection(region1, region2, feature_cols, feature_maxs, feature_mins)
            return_dict[seed] = self.regions_to_tree_improved(train_features_df, train_labels_df, merged_regions, feature_cols,
                                                 feature_mins, feature_maxs, max_samples=max_samples)
            # return 0
        else:
            return_dict[seed] = None
            # return 0

    def genetic_algorithm(self, data, label_col, tree_constructors, population_size=15, num_crossovers=3, val_fraction=0.25,
                          num_iterations=5, seed=1337, tournament_size=3, max_regions=1000, prune=False, max_samples=3,
                          nr_bootstraps=5, mutation_prob=0.25):

        # TODO: use fitness() instead of plain accuracies

        # TODO: use nr_iterations differently: if no improvement made for nr_iteraterions iterations, then stop

        np.random.seed(seed)

        feature_mins = {}
        feature_maxs = {}
        feature_column_names = list(set(data.columns) - set([label_col]))

        for feature in feature_column_names:
            feature_mins[feature] = np.min(data[feature])
            feature_maxs[feature] = np.max(data[feature])

        labels_df = DataFrame()
        labels_df['cat'] = data[label_col].copy()
        features_df = data.copy()
        features_df = features_df.drop(label_col, axis=1)

        data = features_df.copy()
        data[label_col] = labels_df[label_col]

        sss = StratifiedShuffleSplit(labels_df[label_col], 1, test_size=val_fraction, random_state=seed)

        for train_index, test_index in sss:
            train_features_df, test_features_df = features_df.iloc[train_index, :].copy(), features_df.iloc[test_index,
                                                                                           :].copy()
            train_labels_df, test_labels_df = labels_df.iloc[train_index, :].copy(), labels_df.iloc[test_index,
                                                                                     :].copy()
            train_features_df = train_features_df.reset_index(drop=True)
            test_features_df = test_features_df.reset_index(drop=True)
            train_labels_df = train_labels_df.reset_index(drop=True)
            test_labels_df = test_labels_df.reset_index(drop=True)

            smote = SMOTE(ratio='auto', kind='regular')
            # print len(X_train)
            X_train, y_train = smote.fit_sample(train_features_df, train_labels_df)
            train_features_df = DataFrame(X_train, columns=train_features_df.columns)
            train_labels_df = DataFrame(y_train, columns=[label_col])
            perm = np.random.permutation(len(train_features_df))
            train_features_df = train_features_df.iloc[perm].reset_index(drop=True)
            train_labels_df = train_labels_df.iloc[perm].reset_index(drop=True)

            train = data.iloc[train_index, :].copy()
            test = data.iloc[test_index, :].copy()

        tree_list = ISM_v3.bootstrap(train, label_col, tree_constructors, boosting=True, nr_classifiers=nr_bootstraps)
        for constructor in tree_constructors:
            tree = constructor.construct_tree(train_features_df, train_labels_df[label_col])
            print constructor.get_name(), tree.count_nodes()
            tree.populate_samples(train_features_df, train_labels_df[label_col].values)
            tree_list.append(tree)
        for tree in tree_list:
            print tree.count_nodes()
        tree_list = [tree for tree in tree_list if tree is not None ]

        start = time.clock()

        for k in range(num_iterations):
            print "Calculating accuracy and sorting"
            tree_accuracy = []
            for tree in tree_list:
                predicted_labels = tree.evaluate_multiple(test_features_df)
                accuracy = accuracy_score(test_labels_df['cat'].values.astype(str), predicted_labels.astype(str))
                tree_accuracy.append((tree, accuracy, tree.count_nodes()))

            tree_list = [x[0] for x in sorted(tree_accuracy, key=lambda x: (-x[1], x[2]))[:min(len(tree_list), population_size)]]
            print("----> Best tree till now: ", [(x[1], x[2]) for x in sorted(tree_accuracy, key=lambda x: (-x[1], x[2]))[:min(len(tree_list), population_size)]])
            # best_tree_temp = tree_list[0]
            # best_tree_temp.populate_samples(test_features_df, test_labels_df[label_col].values)
            # best_tree_temp.visualise('test')
            # raw_input()

            # Crossovers
            mngr = multiprocessing.Manager()
            return_dict = mngr.dict()
            jobs = []
            for i in range(num_crossovers):
                # new_tree = self.tournament_selection_and_merging(tree_list, train_features_df, train_labels_df,
                #                                                  test_features_df, test_labels_df, label_col,
                #                                                  feature_column_names, feature_maxs, feature_mins,
                #                                                  max_samples, tournament_size)
                p = multiprocessing.Process(target=self.tournament_selection_and_merging, args=[tree_list, train_features_df, train_labels_df,
                                                                     test_features_df, test_labels_df, label_col,
                                                                     feature_column_names, feature_maxs, feature_mins,
                                                                     max_samples, return_dict, k*i+i, tournament_size])
                jobs.append(p)
                p.start()


            for proc in jobs:
                proc.join()

            for new_tree in return_dict.values():
                if new_tree is not None:
                    # new_tree.visualise('test')
                    # raw_input()
                    print 'new tree added', accuracy_score(test_labels_df['cat'].values.astype(str), new_tree.evaluate_multiple(test_features_df).astype(str))
                    tree_list.append(new_tree)

                    if prune:
                        print 'Pruning the tree...', new_tree.count_nodes()
                        new_tree = new_tree.cost_complexity_pruning(train_features_df, train_labels_df['cat'], None, cv=False,
                                                            val_features=test_features_df,
                                                            val_labels=test_labels_df['cat'])
                        print 'Done', new_tree.count_nodes(), accuracy_score(test_labels_df['cat'].values.astype(str), new_tree.evaluate_multiple(test_features_df).astype(str))
                        tree_list.append(new_tree)

            # Mutation phase
            for tree in tree_list:
                value = np.random.rand()
                if value < mutation_prob:
                    new_tree1 = self.mutate_shiftRandom(tree, train_features_df, train_labels_df[label_col].values)
                    print 'new mutation added', accuracy_score(test_labels_df['cat'].values.astype(str),
                                                               new_tree1.evaluate_multiple(test_features_df).astype(str))
                    new_tree2 = self.mutate_swapSubtrees(tree, train_features_df, train_labels_df[label_col].values)
                    print 'new mutation added', accuracy_score(test_labels_df['cat'].values.astype(str),
                                                               new_tree2.evaluate_multiple(test_features_df).astype(str))
                    tree_list.append(new_tree1)
                    tree_list.append(new_tree2)


            end = time.clock()
            print "Took ", (end - start), " seconds"
            start = end

        tree_accuracy = []
        for tree in tree_list:
            predicted_labels = tree.evaluate_multiple(test_features_df)
            accuracy = accuracy_score(test_labels_df[label_col].values.astype(str), predicted_labels.astype(str))
            #print confusion_matrix, accuracy, tree
            tree_accuracy.append((tree, accuracy, tree.count_nodes()))

            # if prune:
            #     print 'Pruning the tree...', tree.count_nodes()
            #     tree = tree.cost_complexity_pruning(train_features_df, train_labels_df['cat'], None, cv=False,
            #                                         val_features=test_features_df, val_labels=test_labels_df['cat'])
            #     predicted_labels = tree.evaluate_multiple(test_features_df)
            #     accuracy = accuracy_score(test_labels_df[label_col].values.astype(str), predicted_labels.astype(str))
            #     print 'Done', tree.count_nodes()
            #
            #     tree_accuracy.append((tree, accuracy, tree.count_nodes()))


        print [x for x in sorted(tree_accuracy, key=lambda x: (-x[1], x[2]))[:min(len(tree_list), population_size)]]

        best_tree = sorted(tree_accuracy, key=lambda x: (-x[1], x[2]))[0][0]
        return best_tree