import copy
import random
from pandas import DataFrame, concat

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import operator

import time

import sklearn
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.utils import resample

from decisiontree import DecisionTree
from featuredescriptors import CONTINUOUS


class LineSegment(object):
    """
        Auxiliary class, used for the intersection algorithm
    """
    def __init__(self, lower_bound, upper_bound, region_index):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.region_index = region_index


class DecisionTreeMerger(object):

    def __init__(self):
        pass

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

    def plot_regions(self, output_path, regions, classes, x_feature, y_feature, x_max=1.0, y_max=1.0, x_min=0.0, y_min=0.0):
        """
        Given an array of 2dimensional regions (classifying 2 classes), having the following format:
            {x_feature: [lb, ub], y_feature: [lb, ub], 'class': {class_1: prob1, class_2: prob2}}
        We return a rectangle divided in these regions, with a purple color according to the class probabilities
        :param output_path: where does the figure need to be saved
        :param regions: the array of regions, according the format described above
        :param classes: the string representations of the 2 possible classes, this is how they are stored in the
                        "class" dictionary of a region
        :param x_feature: what's the string representation of the x_feature in a region?
        :param y_feature: what's the string representation of the y_feature
        :param x_max: maximum value of x_features
        :param y_max: maximum value of y_features
        :param x_min: minimum value of x_features
        :param y_min: minimum value of y_features
        :return: nothing, but saves a figure to output_path
        """
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        plt.axis([x_min, x_max, y_min, y_max])
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        for region in regions:
            if region[x_feature][0] == float("-inf"):
                region[x_feature][0] = x_min
            if region[x_feature][1] == float("inf"):
                region[x_feature][1] = x_max
            if region[y_feature][0] == float("-inf"):
                region[y_feature][0] = y_min
            if region[y_feature][1] == float("inf"):
                region[y_feature][1] = y_max

        for region in regions:
            x = region[x_feature][0]
            width = region[x_feature][1] - x
            y = region[y_feature][0]
            height = region[y_feature][1] - y

            if classes[0] in region['class'] and classes[1] in region['class']:
                purple_tint = (region['class'][classes[0]], 0.0, region['class'][classes[1]])
            elif classes[0] in region['class']:
                purple_tint = (1.0, 0.0, 0.0)
            elif classes[1] in region['class']:
                purple_tint = (0.0, 0.0, 1.0)
            else:
                print "this shouldn't have happened, go look at treemerger.py"

            ax1.add_patch(
                patches.Rectangle(
                    (x, y),   # (x,y)
                    width,          # width
                    height,          # height
                    facecolor=purple_tint
                )
            )

        fig1.savefig(output_path)

    def plot_regions_with_points(self, output_path, regions, classes, x_feature, y_feature, points, x_max=1.0, y_max=1.0, x_min=0.0, y_min=0.0):
        """
        Given an array of 2dimensional regions (classifying 2 classes), having the following format:
            {x_feature: [lb, ub], y_feature: [lb, ub], 'class': {class_1: prob1, class_2: prob2}}
        We return a rectangle divided in these regions, with a purple color according to the class probabilities
        :param output_path: where does the figure need to be saved
        :param regions: the array of regions, according the format described above
        :param classes: the string representations of the 2 possible classes, this is how they are stored in the
                        "class" dictionary of a region
        :param x_feature: what's the string representation of the x_feature in a region?
        :param y_feature: what's the string representation of the y_feature
        :param x_max: maximum value of x_features
        :param y_max: maximum value of y_features
        :param x_min: minimum value of x_features
        :param y_min: minimum value of y_features
        :return: nothing, but saves a figure to output_path
        """
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        plt.axis([x_min, x_max, y_min, y_max])
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        for region in regions:
            x = region[x_feature][0]
            width = region[x_feature][1] - x
            y = region[y_feature][0]
            height = region[y_feature][1] - y

            if classes[0] in region['class'] and classes[1] in region['class']:
                purple_tint = (region['class'][classes[0]], 0.0, region['class'][classes[1]])
            elif classes[0] in region['class']:
                purple_tint = (1.0, 0.0, 0.0)
            elif classes[1] in region['class']:
                purple_tint = (0.0, 0.0, 1.0)
            else:
                print "this shouldn't have happened, go look at treemerger.py"

            ax1.add_patch(
                patches.Rectangle(
                    (x, y),   # (x,y)
                    width,          # width
                    height,          # height
                    facecolor=purple_tint
                )
            )


        for i in range(len(points.index)):
            x = points.iloc[i][x_feature]
            y = points.iloc[i][y_feature]
            ax1.add_patch(
                patches.Circle(
                    (x, y),   # (x,y)
                    0.001,          # width
                    facecolor='black'
                )
            )

        fig1.savefig(output_path)

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

        # The intersection of all these S_i's are the intersecting regions
        intersection_regions_indices = S_intersections[0]
        for k in range(1, len(S_intersections)):
            intersection_regions_indices = self.tuple_list_intersections(intersection_regions_indices, S_intersections[k])

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

    def calculate_entropy(self, values_list):
        if sum(values_list) == 0:
            return 0
        # Normalize the values by dividing each value by the sum of all values in the list
        normalized_values = map(lambda x: float(x) / float(sum(values_list)), values_list)

        # Calculate the log of the normalized values (negative because these values are in [0,1])

        log_values = map(lambda x: np.log(x)/np.log(2), normalized_values)

        # Take sum of normalized_values * log_values, multiply with (-1) to get positive float
        return -sum(np.multiply(normalized_values, log_values))

    def split_criterion(self, node):
        """
        Calculates information gain ratio (ratio because normal gain tends to have a strong bias in favor of tests with
        many outcomes) for a subtree
        :param node: the node where the information gain needs to be calculated for
        :return: the information gain: information (entropy) in node - sum(weighted information in its children)
        """
        counts_before_split = np.asarray(node.data[['cat', node.label]].groupby(['cat']).count().values[:, 0])
        total_count_before_split = sum(counts_before_split)
        info_before_split = self.calculate_entropy(counts_before_split)

        if len(node.left.data[['cat', node.label]].groupby(['cat']).count().values) > 0:
            left_counts = np.asarray(node.left.data[['cat', node.label]].groupby(['cat']).count().values[:, 0])
        else:
            left_counts = [0]
        total_left_count = sum(left_counts)
        if len(node.right.data[['cat', node.label]].groupby(['cat']).count().values) > 0:
            right_counts = np.asarray(node.right.data[['cat', node.label]].groupby(['cat']).count().values[:, 0])
        else:
            right_counts = [0]
        total_right_count = sum(right_counts)

        # Information gain after split = weighted entropy of left child + weighted entropy of right child
        # weight = number of nodes in child / sum of nodes in left and right child
        info_after_split = float(total_left_count) / float(total_count_before_split) * self.calculate_entropy(left_counts) \
                           + float(total_right_count) / float(total_count_before_split) * self.calculate_entropy(right_counts)

        gini_coeff = (float(total_left_count) / float(total_count_before_split)) * \
                     (float(total_right_count) / float(total_count_before_split))

        return ((info_before_split - info_after_split) / info_before_split) + gini_coeff

    def divide_data(self, data, feature, value):
            """
            Divide the data in two subsets, thanks pandas
            :param data: the dataframe to divide
            :param feature: on which column of the dataframe are we splitting?
            :param value: what threshold do we use to split
            :return: node: initialised decision tree object
            """
            return DecisionTree(left=DecisionTree(data=data[data[feature] <= value]),
                                right=DecisionTree(data=data[data[feature] > value]),
                                label=feature,
                                data=data,
                                value=value)

    def dec(self, input_, output_):
        if type(input_) is list:
            for subitem in input_:
                self.dec(subitem, output_)
        else:
            output_.append(input_)

    def generate_samples(self, regions, features, feature_descriptors):
        columns = list(features)
        columns.append('cat')
        samples = DataFrame(columns=columns)
        print "Generating samples for ", len(regions), " regions"
        counter = 0
        for region in regions:
            counter += 1
            region_copy = region.copy()
            del region_copy['class']
            sides = region_copy.items()

            sides = sorted(sides, key=lambda x: x[1][1] - x[1][0])

            amount_of_samples = 1
            for side in sides[:int(math.ceil(np.sqrt(len(features))))]:
                if side[1][1] - side[1][0] > 0:
                    amount_of_samples += (side[1][1] - side[1][0]) * 50

            if len(region['class']) > 0:
                amount_of_samples *= max(region['class'].iteritems(), key=operator.itemgetter(1))[1]
            else:
                amount_of_samples = 0

            print "----> Region ", counter, ": ", int(amount_of_samples), " samples"

            point = {}
            for feature_index in range(len(features)):
                if feature_descriptors[feature_index][0] == CONTINUOUS:
                    point[features[feature_index]] = region[features[feature_index]][0] + \
                                                      ((region[features[feature_index]][1] - region[features[feature_index]][0]) / 2)

            for k in range(int(amount_of_samples)):
                for feature_index in range(len(features)):
                    if feature_descriptors[feature_index][0] == CONTINUOUS:
                        if region[features[feature_index]][1] - region[features[feature_index]][0] > 1.0:
                            point[features[feature_index]] += (random.random() - 0.5) * \
                                                              np.sqrt((region[features[feature_index]][1] - region[features[feature_index]][0]))
                        else:
                            point[features[feature_index]] += (random.random() - 0.5) * \
                                pow((region[features[feature_index]][1] - region[features[feature_index]][0]), 2)
                    else:
                        choice_list = np.arange(region[features[feature_index]][0], region[features[feature_index]][1],
                                                1.0/float(feature_descriptors[feature_index][1])).tolist()
                        if len(choice_list) > 0:
                            choice_list.extend([region[features[feature_index]][0] + (region[features[feature_index]][1] -
                                                                                      region[features[feature_index]][0])/2]*len(choice_list)*2)
                            point[features[feature_index]] = random.choice(choice_list)
                        else:
                            point[features[feature_index]] = region[features[feature_index]][0]

                point['cat'] = max(region['class'].iteritems(), key=operator.itemgetter(1))[0]
                samples = samples.append(point, ignore_index=True)

        return samples


    def evaluate_regions(self, regions, test_features_df, default=1):
        #print "Evaluating regions..."
        labels = []
        for i in range(len(test_features_df)):
            sample = test_features_df.iloc[i, :].to_dict()
            features = sample.keys()
            for region in regions:
                found = True
                #print features, sample, region
                for feature in features:
                    if not(region[feature][0] <= sample[feature] and region[feature][1] >= sample[feature]):
                        found = False
                if found:
                    labels.append(max(region['class'].iteritems(), key=operator.itemgetter(1))[0])
                    break
            if len(labels) != i+1:
                labels.append(default)
        return np.asarray(labels)

    def find_lines(self, regions, features, feature_mins, feature_maxs):
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
            lines[feature] = np.unique(lower_upper_regions[lower_upper_regions.duplicated(feature+'_lb', False)][feature+'_lb'])

        return lines

    def calculate_info_gains(self, lines, features_df, labels_df):
        data = DataFrame(features_df)
        data['cat'] = labels_df
        info_gains = {}
        for key in lines:
            for value in lines[key]:
                node = self.divide_data(data, key, value)
                split_crit = self.split_criterion(node)
                if split_crit > 0:
                    info_gains[node] = self.split_criterion(node)
        return info_gains


    def regions_to_tree_improved(self, features_df, labels_df, regions, features, feature_mins, feature_maxs, max_samples=5):
        lines = self.find_lines(regions, features, feature_mins, feature_maxs)
        # print "Found ", len(lines), " lines for ", len(regions), " regions and ", len(features), " features in ", (end-start), " seconds"
        # print lines

        if lines is None or len(lines) <= 0:
            return DecisionTree(label=str(np.argmax(np.bincount(labels_df['cat'].values.astype(int)))), value=None, data=features_df)

        info_gains = self.calculate_info_gains(lines, features_df, labels_df)
        # print "Calculated info gains ", len(lines), " features and ", len(features_df), " samples in ", (end-start), " seconds"
        # print info_gains

        if len(info_gains) > 0:
            best_split_node = max(info_gains.items(), key=operator.itemgetter(1))[0]
            node = DecisionTree(label=best_split_node.label, value=best_split_node.value, data=best_split_node.data)
        else:
            node = DecisionTree(label=str(np.argmax(np.bincount(labels_df['cat'].values.astype(int)))), data=features_df, value=None)
            return node
        # print node.label, node.value

        ##########################################################################

        # We call recursively with the splitted data and set the bounds of feature f
        # for left child: set upper bounds to the value of the chosen line
        # for right child: set lower bounds to value of chosen line
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
        if len(regions_left) >= max_samples or len(regions_right) >= max_samples:
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

    def genetic_algorithm(self, data, cat_name, tree_constructors, population_size=15, num_mutations=3,
                          val_fraction=0.25, num_iterations=5, seed=1337, max_samples=5, num_boosts=3,
                          max_regions=1000, prune=False):

        print "Initializing"
        #################################
        #       Initialisation          #
        #################################
        np.random.seed(seed)

        feature_mins = {}
        feature_maxs = {}
        feature_column_names = list(set(data.columns) - set([cat_name]))

        for feature in feature_column_names:
                feature_mins[feature] = np.min(data[feature])
                feature_maxs[feature] = np.max(data[feature])

        ###############################################
        #      Boosting to create population          #
        ###############################################
        labels_df = DataFrame()
        labels_df['cat'] = data[cat_name].copy()
        features_df = data.copy()
        features_df = features_df.drop(cat_name, axis=1)

        sss = StratifiedShuffleSplit(labels_df['cat'], 1, test_size=val_fraction, random_state=seed)

        for train_index, test_index in sss:
            train_features_df, test_features_df = features_df.iloc[train_index, :].copy(), features_df.iloc[test_index, :].copy()
            train_labels_df, test_labels_df = labels_df.iloc[train_index, :].copy(), labels_df.iloc[test_index, :].copy()
            train_features_df = train_features_df.reset_index(drop=True)
            test_features_df = test_features_df.reset_index(drop=True)
            train_labels_df = train_labels_df.reset_index(drop=True)
            test_labels_df = test_labels_df.reset_index(drop=True)

        #TODO: populate start_trees by boosting
        start_trees = []
        start_regions = []
        regions_list = []

        for tree_constructor in tree_constructors:
            boosting_train_features_df = train_features_df.copy()
            boosting_train_labels_df = train_labels_df.copy()
            regions_to_merge = []
            tree = tree_constructor.construct_tree(train_features_df, train_labels_df['cat'])
            tree.populate_samples(train_features_df, train_labels_df['cat'])
            regions = self.decision_tree_to_decision_table(tree, train_features_df)
            regions_list.append(regions)
            start_trees.append(tree)
            start_regions.append(regions)
            regions_to_merge.append(regions)
            for i in range(num_boosts-1):
                missclassified_features = []
                missclassified_labels = []
                for i in range(len(train_features_df)):
                    predicted_label = tree.evaluate(train_features_df.iloc[i, :])
                    real_label = train_labels_df.iloc[i, :]['cat']
                    if real_label != predicted_label:
                        missclassified_features.append(train_features_df.iloc[i, :])
                        missclassified_labels.append(train_labels_df.iloc[i, :])

                boosting_train_features_df = concat([DataFrame(missclassified_features), boosting_train_features_df])
                boosting_train_labels_df = concat([DataFrame(missclassified_labels), boosting_train_labels_df])
                boosting_train_features_df = boosting_train_features_df.reset_index(drop=True)
                boosting_train_labels_df = boosting_train_labels_df.reset_index(drop=True)

                tree = tree_constructor.construct_tree(boosting_train_features_df, boosting_train_labels_df)
                tree.populate_samples(boosting_train_features_df, boosting_train_labels_df['cat'])
                regions = self.decision_tree_to_decision_table(tree, boosting_train_features_df)
                regions_list.append(regions)
                start_trees.append(tree)
                regions_to_merge.append(regions)

            if num_boosts > 1:
                merged_regions = self.calculate_intersection(regions_to_merge[0], regions_to_merge[1], feature_column_names,
                                                             feature_maxs, feature_mins)
                for k in range(2, len(regions_to_merge)):
                    if len(merged_regions) < max_regions:
                        merged_regions = self.calculate_intersection(merged_regions, regions_to_merge[k],
                                                                     feature_column_names, feature_maxs, feature_mins)
                    else:
                        break
                regions_list.append(merged_regions)

        if num_boosts == 1:
            merged_regions = self.calculate_intersection(regions_list[0], regions_list[1], feature_column_names,
                                                             feature_maxs, feature_mins)
            for k in range(2, len(regions_list)):
                if len(merged_regions) < max_regions:
                    merged_regions = self.calculate_intersection(merged_regions, regions_list[k],
                                                                 feature_column_names, feature_maxs, feature_mins)
                else:
                    break
            regions_list.append(merged_regions)

        # for tree_constructor in tree_constructors:
        #     tree = tree_constructor.construct_tree(train_features_df, train_labels_df)
        #     tree.populate_samples(train_features_df, train_labels_df['cat'])
        #     regions = self.decision_tree_to_decision_table(tree, train_features_df)
        #     regions_list.append(regions)
        #     start_trees.append(tree)
        #     start_regions.append(regions)

        ###############################################
        #           The genetic algorithm             #
        ###############################################

        start = time.clock()
        for k in range(num_iterations):

            # For each class, and each possible tree, calculate their respective class accuracy
            print "Calculating accuracy and sorting"
            tree_accuracy = []
            for region in regions_list:
                predicted_labels = self.evaluate_regions(region, test_features_df)
                confusion_matrix = DecisionTree.plot_confusion_matrix(test_labels_df['cat'].values.astype(str), predicted_labels.astype(str))
                confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum()
                accuracy = sum([confusion_matrix[i][i] for i in range(len(confusion_matrix))])
                tree_accuracy.append((region, accuracy))

            # Pick the |population_size| best trees
            regions_list = [x[0] for x in sorted(tree_accuracy, key=operator.itemgetter(1), reverse=True)[:min(len(regions_list), population_size)]]
            print("----> Best tree till now: ", [x[1] for x in sorted(tree_accuracy, key=operator.itemgetter(1), reverse=True)[:min(len(regions_list), population_size)]])
            # Breeding phase: we pick one tree from each of the top class predictor sets and merge them all together
            # We create the new constructed_trees array and regions list dict for the next iteration
            trees_to_merge = range(min(len(regions_list), num_mutations*2))
            while len(trees_to_merge) > 1:
                print("breeding.... ", len(trees_to_merge))
                indexA = np.random.choice(trees_to_merge)
                trees_to_merge.remove(indexA)
                indexB = np.random.choice(trees_to_merge)
                trees_to_merge.remove(indexB)
                print("merging...")
                merged_regions = self.calculate_intersection(regions_list[indexA], regions_list[indexB],
                                                             feature_column_names, feature_maxs, feature_mins)
                print("going to tree")
                predicted_labels = self.evaluate_regions(merged_regions, test_features_df)
                confusion_matrix = DecisionTree.plot_confusion_matrix(test_labels_df[cat_name].values.astype(str), predicted_labels.astype(str))
                confusion_matrix = np.around(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis], 3)
                print sum([confusion_matrix[i][i] for i in range(len(confusion_matrix))])

                if len(merged_regions) < max_regions:
                    regions_list.append(merged_regions)

            end = time.clock()
            print "Took ", (end - start), " seconds"
            start = end

        regions_list = [x[0] for x in sorted(tree_accuracy, key=operator.itemgetter(1), reverse=True)[:min(len(regions_list), population_size)]]

        # Now take the best trees to return
        print "Taking best tree.."
        tree_accuracy = []
        counter = 1

        region_accuracy = []

        # for region in regions_list:
        #     predicted_labels = self.evaluate_regions(region, test_features_df)
        #     confusion_matrix = tree.plot_confusion_matrix(test_labels_df[cat_name].values.astype(str), predicted_labels.astype(str))
        #     confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum()
        #     accuracy = sum([confusion_matrix[i][i] for i in range(len(confusion_matrix))])
        #     print confusion_matrix, accuracy
        #     region_accuracy.append((region, accuracy, len(region)))

        for region in regions_list:
                print counter, len(region)
                tree = self.regions_to_tree_improved(train_features_df, train_labels_df, region, feature_column_names,
                                                     feature_mins, feature_maxs, max_samples=max_samples)
                if prune:
                    print 'Pruning the tree...', tree.count_nodes()
                    tree = tree.cost_complexity_pruning(train_features_df, train_labels_df['cat'], None, cv=False,
                                                        val_features=test_features_df, val_labels=test_labels_df['cat'])
                    print 'Done', tree.count_nodes()
                predicted_labels = tree.evaluate_multiple(test_features_df)
                confusion_matrix = tree.plot_confusion_matrix(test_labels_df[cat_name].values.astype(str), predicted_labels.astype(str))
                confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum()

                accuracy = sum([confusion_matrix[i][i] for i in range(len(confusion_matrix))])
                #print confusion_matrix, accuracy, tree
                tree_accuracy.append((tree, accuracy, len(region)))
                counter += 1

        for tree in start_trees:
            # eature_vectors, labels, tree_constructor, ism_constructors = [],
            # ism_calc_fracs = False, ism_nr_classifiers = 3, ism_boosting = False, n_folds = 3,
            # cv = True, val_features = None, val_labels = None)
            if prune:
                print 'Pruning the tree...', tree.count_nodes()
                tree = tree.cost_complexity_pruning(train_features_df, train_labels_df['cat'], None, cv=False,
                                                    val_features=test_features_df, val_labels=test_labels_df['cat'])
                print 'Done', tree.count_nodes()
            predicted_labels = tree.evaluate_multiple(test_features_df)
            confusion_matrix = tree.plot_confusion_matrix(test_labels_df[cat_name].values.astype(str), predicted_labels.astype(str))
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum()
            accuracy = sum([confusion_matrix[i][i] for i in range(len(confusion_matrix))])
            #print confusion_matrix, accuracy, tree
            tree_accuracy.append((tree, accuracy, len(regions)))
            # region = self.decision_tree_to_decision_table(tree, train_features_df)
            # predicted_labels = self.evaluate_regions(regions, test_features_df)
            # confusion_matrix = tree.plot_confusion_matrix(test_labels_df[cat_name].values.astype(str), predicted_labels.astype(str))
            # confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum()
            # accuracy = sum([confusion_matrix[i][i] for i in range(len(confusion_matrix))])
            # print confusion_matrix, accuracy
            # region_accuracy.append((region, accuracy, len(region)))

        print [x for x in sorted(tree_accuracy, key=lambda x: (-x[1], x[2]))[:min(len(regions_list), population_size)]]

        best_tree = sorted(tree_accuracy, key=lambda x: (-x[1], x[2]))[0][0]
        #best_region = sorted(region_accuracy, key=lambda x: (-x[1], x[2]))[0][0]

        return best_tree#(best_tree, best_region)

