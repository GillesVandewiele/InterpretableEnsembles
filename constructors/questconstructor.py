"""
    Written by Kiani Lannoye & Gilles Vandewiele
    Commissioned by UGent.

    Design of a diagnose- and follow-up platform for patients with chronic headaches
"""

from pandas import DataFrame
import pandas as pd

import math
from sklearn.cluster import k_means
from sklearn.feature_selection import chi2, f_classif

import numpy as np

from constructors.treeconstructor import TreeConstructor
from decisiontree import DecisionTree

CONTINUOUS = "continuous"
DISCRETE = "discrete"

class QuestConstructor(TreeConstructor):
    """
    Contains our own implementation of the QUEST algorithm. The algorithm can be found on:
    ftp://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/13.0/TREE-QUEST.pdf
    """

    def __init__(self, default=1, max_nr_nodes=1, discrete_thresh=5, alpha=0.1, max_depth=20):
        self.default = default
        self.max_nr_nodes = max_nr_nodes
        self.discrete_thresh = discrete_thresh
        self.alpha = alpha
        self.max_depth=max_depth

    def get_name(self):
        return "QUEST"

    def all_feature_vectors_equal(self, training_feature_vectors):
        return len(training_feature_vectors.index) == (training_feature_vectors.duplicated().sum() + 1)

    def levene_f_test(self, data):
        # For each feature and each class, calculate the mean per class
        feature_columns = data.columns[:-1]
        unique_categories = np.unique(data['cat'])
        mean_per_feature_and_class = {}
        for feature in feature_columns:
            feature_mean_per_class = {}
            for category in unique_categories:
                data_feature_cat = data[(data.cat == category)][feature]
                feature_mean_per_class[category] = float(sum(data_feature_cat)/len(data_feature_cat))
            mean_per_feature_and_class[feature] = feature_mean_per_class

        # Then tranform all the data (sample_point - mean)
        for feature in feature_columns:
            data[feature] = data[[feature, 'cat']].apply((lambda x: abs(x[0] - mean_per_feature_and_class[feature][x[1]])), axis=1)

        return f_classif(data[feature_columns], np.ravel(data['cat']))

    def divide_data(self, data, feature, value):
        """
        Divide the data in two subsets, thanks pandas
        :param data: the dataframe to divide
        :param feature: on which column of the dataframe are we splitting?
        :param value: what threshold do we use to split
        :return: node: initialised decision tree object
        """
        # print data[feature], feature, value
        return DecisionTree(left=DecisionTree(data=data[data[feature] <= value]),
                            right=DecisionTree(data=data[data[feature] > value]),
                            label=feature,
                            data=data,
                            value=value)

    def construct_tree(self, training_feature_vectors, labels, current_depth=0):
        # First find the best split feature
        feature, type = self.find_split_feature(training_feature_vectors.copy(), labels.copy())

        # Can be removed later
        if len(labels) == 0:
            return DecisionTree(label=self.default, value=None, data=None)

        data = DataFrame(training_feature_vectors.copy())
        data['cat'] = labels

        # Only pre-pruning enabled at this moment (QUEST already has very nice trees)
        if feature is None or len(data) == 0 or len(training_feature_vectors.index) <= self.max_nr_nodes \
                or len(np.unique(data['cat'])) == 1 or self.all_feature_vectors_equal(training_feature_vectors)\
                or current_depth >= self.max_depth:
            # Create leaf with label most occurring class
            label = np.argmax(np.bincount(data['cat'].values.astype(int)))
            return DecisionTree(label=label.astype(str), value=None, data=data)

        # If we don't need to pre-prune, we calculate the best possible splitting point for the best split feature
        split_point = self.find_best_split_point(data.copy(), feature, type)

        if split_point is None or math.isnan(split_point):
            label = np.argmax(np.bincount(data['cat'].values.astype(int)))
            return DecisionTree(label=label.astype(str), value=None, data=data)


        # Divide the data using this best split feature and value and call recursively
        split_node = self.divide_data(data.copy(), feature, split_point)
        if len(split_node.left.data) == 0 or len(split_node.right.data) == 0:
            label = np.argmax(np.bincount(data['cat'].values.astype(int)))
            return DecisionTree(label=label.astype(str), value=None, data=data)
        node = DecisionTree(label=split_node.label, value=split_node.value, data=split_node.data)
        node.left = self.construct_tree(split_node.left.data.drop('cat', axis=1),
                                        split_node.left.data[['cat']], current_depth+1)
        node.right = self.construct_tree(split_node.right.data.drop('cat', axis=1),
                                         split_node.right.data[['cat']], current_depth+1)

        return node

    def find_split_feature(self, training_feature_vectors, labels):
        """
        Find the best possible feature to split on (the value to split on will be calculated further)
        :param training_feature_vectors: a pandas dataframe containing the features
        :param labels: a pandas dataframe containing the labels in the same order
        :return: decision_tree: a DecisionTree object
        """

        if len(labels) == 0:
            return None, None
        cols = training_feature_vectors.columns

        data = DataFrame(training_feature_vectors.copy())
        data['cat'] = labels

        # Split dataframe in continuous and discrete features:
        continuous_features = []
        discrete_features = []
        for feature in cols.values:
            if len(np.unique(data[feature])) > self.discrete_thresh:
                continuous_features.append(feature)
            else:
                discrete_features.append(feature)

        # For discrete features, we calculate chi^2 scores (p-value)
        if len(discrete_features) > 0:
            chi2_scores, chi2_values = chi2(training_feature_vectors[discrete_features], np.ravel(labels.values))
        else:
            chi2_values = []

        # For continuous ones, we calculate anova f scores (p-value)
        if len(continuous_features) > 0:
            anova_f_scores, anova_f_values = f_classif(training_feature_vectors[continuous_features],
                                                       np.ravel(labels.values))
        else:
            anova_f_values = []

        # Remove nan's and pick the feature with the lowest score
        chi2_values = np.where(np.isnan(chi2_values), 1, chi2_values)
        anova_f_values = np.where(np.isnan(anova_f_values), 1, anova_f_values)
        conc = np.concatenate([chi2_values, anova_f_values])
        conc_features = np.concatenate([discrete_features, continuous_features])
        best_feature_p_value = min(conc)
        best_feature = conc_features[np.argmin(conc)]

        # If the p-value is smaller than a weighted threshold alpha, we found a feature
        if best_feature_p_value < self.alpha/len(cols.values):
            if best_feature in continuous_features:
                return best_feature, CONTINUOUS
            else:
                return best_feature, DISCRETE
        else:
            # Else, we apply levene f test (which is the same as anova, but with a conversion of
            # the values as pre-process step)
            continuous_features_cat = [item for sublist in [continuous_features, ["cat"]] for item in sublist]
            if len(continuous_features) == 0:
                return None, None
            levene_scores, levene_values = self.levene_f_test(data[continuous_features_cat].copy())
            best_feature_p_value = min(levene_values)
            best_feature = continuous_features[np.argmin(levene_values)]
            if best_feature_p_value < self.alpha/(len(cols.values)+len(continuous_features)):
                if best_feature in continuous_features:
                    return best_feature, CONTINUOUS
                else:
                    return best_feature, DISCRETE
            else:
                return None, None

    def find_best_split_point(self, data, feature, type):
        unique_categories = np.unique(data['cat'])
        feature_mean_var_freq_per_class = []
        max_value = []
        min_value = []

        for category in unique_categories:
            data_feature_cat = data[(data.cat == category)][feature]
            feature_mean_var_freq_per_class.append([float(np.mean(data_feature_cat)), float(np.var(data_feature_cat)),
                                                     len(data_feature_cat), category])
            max_value.append(np.max(data_feature_cat.astype(int)))
            min_value.append(np.min(data_feature_cat.astype(int)))

        max_value = np.max(max_value)
        min_value = np.min(min_value)

        # First we transform nominal discrete variables to a continuous variable and then apply same QDA
        if type == DISCRETE:  # and False:  #TODO: check if it is DISCRETE and NOMINAL
            data_feature_all_cats = pd.get_dummies(data[feature])
            dummies = data_feature_all_cats.columns
            data_feature_all_cats['cat'] = data['cat']
            mean_freq_per_class_dummies = []
            for category in unique_categories:
                data_feature_cat = data_feature_all_cats[(data.cat == category)]
                mean_freq_per_class_dummies.append([data_feature_cat.as_matrix(columns=dummies).mean(0),
                                                    len(data_feature_cat.index)])
            overall_mean = data_feature_all_cats.as_matrix(columns=dummies).mean(0)

            split_point = 0
            # For each class we construct an I x I matrix (with I number of variables), some reshaping required
            B_temp = ([np.dot(np.transpose(np.reshape(np.subtract(mean_freq_per_class_dummies[i][0], overall_mean), (1, -1))),
                         np.reshape(np.subtract(mean_freq_per_class_dummies[i][0], overall_mean), (1, -1)))
                  for i in range(len(mean_freq_per_class_dummies))])
            # B_temp = ([np.dot(np.reshape(np.subtract(mean_freq_per_class_dummies[i][0], overall_mean), (1, -1)),
            #              np.transpose(np.reshape(np.subtract(mean_freq_per_class_dummies[i][0], overall_mean), (1, -1))))
            #       for i in range(len(mean_freq_per_class_dummies))])
            B = B_temp[0]
            for i in range(1, len(B_temp)):
                B = np.add(B, B_temp[i])

            T_temp = [np.dot(np.transpose(np.reshape(np.subtract(data_feature_all_cats.as_matrix(columns=dummies)[i,:],
                                                                 overall_mean), (1, -1))),
                             np.reshape(np.subtract(data_feature_all_cats.as_matrix(columns=dummies)[i, :],
                                                    overall_mean), (1, -1)))
                      for i in range(len(data_feature_all_cats.index))]
            # T_temp = [np.dot(np.reshape(np.subtract(data_feature_all_cats.as_matrix(columns=dummies)[i,:],
            #                                                      overall_mean), (1, -1)),
            #                  np.transpose(np.reshape(np.subtract(data_feature_all_cats.as_matrix(columns=dummies)[i, :],
            #                                         overall_mean), (1, -1))))
            #           for i in range(len(data_feature_all_cats.index))]
            T = T_temp[0]
            for i in range(1, len(T_temp)):
                T = np.add(T, T_temp[i])

            # Perform single value decomposition on T: T = Q*D*Q'
            Q, D, Q_t = np.linalg.svd(T)
            # Make sure we don't do sqrt of negative numbers
            D = [0 if i < 0 else 1/i for i in D]
            D_sqrt = np.sqrt(D)
            # Make sure we don't invert zeroes
            D_sqrt_inv = np.diag([0 if i == 0 else 1/i for i in D_sqrt])

            # Get most important eigenvector of using D
            matrix = np.dot(np.dot(np.dot(np.dot(D_sqrt_inv, Q_t), B), Q), D_sqrt_inv)
            if np.isnan(matrix).any():
                return None
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            largest_eigenvector = eigenvectors[np.argmax(eigenvalues)]

            # We can now transform all discrete attributes to continous ones!
            discrete_values = data[feature].values
            continous_values = []
            discrete_dummies = data_feature_all_cats.loc[:,dummies].values.astype(float)
            for i in range(len(discrete_values)):
                new_value = np.dot(np.dot(np.dot(np.reshape(largest_eigenvector, (1, -1)), D_sqrt_inv), Q_t), discrete_dummies[i])
                continous_values.append(new_value[0])
            data[feature] = continous_values

        # Now find the best split point
        means = [i[0] for i in feature_mean_var_freq_per_class]
        variances = [i[1] for i in feature_mean_var_freq_per_class]
        frequencies = [i[2] for i in feature_mean_var_freq_per_class]
        if len(unique_categories) != 2:
            # If all class means are equal, pick the one with highest frequency as superclass A, the others as B
            # Then calculate means and variances of the two superclasses
            if len(np.unique(means)) == 1:
                index_a = np.argmax(frequencies)
                mean_a = means[index_a]
                var_a = variances[index_a]
                freq_a = frequencies[index_a]
                sum_freq = sum(frequencies)
                mean_b = sum([(frequencies[i]*means[i])/sum_freq for i in range(len(means)) if i != index_a])
                var_b = sum([(frequencies[i]*variances[i] + frequencies[i]*(means[i] - mean_b))/sum_freq for i in range(len(means)) if i != index_a])
                freq_b = sum([frequencies[i] for i in range(len(means)) if i != index_a])
                #print(mean_a, mean_b, mean_b, variance_b)

            # Else, apply kmeans clustering to divide the classes in 2 superclasses
            else:
                clusters = k_means(np.reshape(means, (-1, 1)), 2,
                                            n_init=1, init=np.asarray([[np.min(means)], [np.max(means)]]))
                mean_a, mean_b = np.ravel(clusters[0])
                labels = clusters[1]
                indices_a = [i for i in range(len(labels)) if labels[i] == 0]
                indices_b = [i for i in range(len(labels)) if labels[i] == 1]
                sum_freq = sum(frequencies)
                var_a = sum([(frequencies[i]*(variances[i] + (means[i] - mean_a)**2)) for i in indices_a])/sum_freq
                var_b = sum([(frequencies[i]*(variances[i] + (means[i] - mean_b)**2)) for i in indices_b])/sum_freq
                freq_a = sum([frequencies[i] for i in indices_a])
                freq_b = sum([frequencies[i] for i in indices_b])

               # print([mean_a, mean_b], [var_a, var_b], [freq_a, freq_b])

        # If there are only two classes, those are the superclasses already
        else:
            mean_a = means[0]
            mean_b = means[1]
            var_a = variances[0]
            var_b = variances[1]
            freq_a = frequencies[0]
            freq_b = frequencies[1]

        split_point = self.calculate_split_point(mean_a, mean_b, var_a, var_b, freq_a, freq_b, max_value, min_value)

        return split_point

    def calculate_split_point(self, mean_a, mean_b, var_a, var_b, freq_a, freq_b, max_val, min_val):
        if np.min([var_a, var_b]) == 0.0:
            if var_a < var_b:
                if mean_a < mean_b:
                    return mean_a*(1+10**(-12))
                else:
                    return mean_a*(1-10**(-12))
            else:
                if mean_b < mean_a:
                    return mean_b*(1+10**(-12))
                else:
                    return mean_b*(1-10**(-12))
        # Else quadratic discriminant analysis (find X such that P(X, A|t) = P(X, B|t))
        else:
            a = var_a - var_b
            b = 2*(mean_a*var_b - mean_b*var_a)
            prob_a = float(float(freq_a) / float(freq_a + freq_b))
            prob_b = 1 - prob_a
            c = (mean_b**2)*var_a - (mean_a**2)*var_b + 2*var_a*var_b*np.log2((prob_a * np.sqrt(var_b))/(prob_b * np.sqrt(var_a)))

            disc = b**2-4*a*c
            if disc == 0:
                x1 = (-b+np.sqrt(disc))/(2*a)
                if x1 < min_val:
                    return min_val
                elif x1 > max_val:
                    return (mean_a + mean_b)/2
                else:
                    return x1
            elif disc > 0:
                x1 = (-b+np.sqrt(disc))/(2*a)
                x2 = (-b-np.sqrt(disc))/(2*a)
                if abs(x1 - mean_a) < abs(x2 - mean_a):
                    if x1 < min_val:
                        return min_val
                    elif x1 > max_val:
                        return max_val
                    else:
                        return x1
                else:
                    if x2 < min_val:
                        return min_val
                    elif x2 > max_val:
                        return max_val
                    else:
                        return x2
            else:
                return (mean_a + mean_b)/2

