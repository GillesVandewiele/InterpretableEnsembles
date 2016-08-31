"""
    Interpretable Single Model
    --------------------------

    Merges different decision trees in an ensemble together in a single, interpretable decision tree

    written by Gilles Vandewiele

    Van Assche, Anneleen, and Hendrik Blockeel.
    "Seeing the forest through the trees: Learning a comprehensible model from an ensemble."
    European Conference on Machine Learning. Springer Berlin Heidelberg, 2007.
"""

# External imports
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Internal imports
from decisiontree import DecisionTree


def extract_tests(tree, tests=set()):
    """
    Given a decision tree, extract all tests from the nodes

    :param tree: the decision tree to extract tests from (decisiontree.py)
    :param tests: recursive parameter, don't touch
    :return: a set of possible tests (feature_label <= threshold_value); each entry is a tuple (label, value)
    """
    if tree.value is not None:
        tests.add((tree.label, tree.value))
        extract_tests(tree.left, tests)
        extract_tests(tree.right, tests)
    return tests


def calculate_entropy(probabilities):
    """
    Calculate the entropy of given probabilities

    :param probabilities: a list of floats between [0, 1] (sum(probabilities) must be 1)
    :return: the entropy
    """
    return sum([-prob * np.log(prob)/np.log(2) if prob != 0 else 0 for prob in probabilities])


def get_most_occurring_class(data, class_label):
    """
    Get the most occurring class in a dataframe of data

    :param data: a pandas dataframe
    :param class_label: the column of the class labels
    :return: the most occurring class
    """
    return Counter(data[class_label].values.tolist()).most_common(1)[0][0]


def calculate_prob(tree, label, value, prior_tests, negate=False):
    """
    Estimate the probabilities from a decision tree by propagating down from the root to the leaves

    :param tree: the decision tree to estimate the probabilities from
    :param label: the label of the test being evaluated
    :param value: the value of the test being evaluated
    :param prior_tests: tests that are already in the conjunctions
    :param negate: is it a negative or positive test
    :return: a vector of probabilities for each class
    """
    if tree.value is None:  # If the value is None, we're at a leaf, return a vector of probabilities
        return np.divide(list(map(float, tree.class_probabilities.values())), float(sum(tree.class_probabilities.values())))
    else:
        if (tree.label, tree.value) in prior_tests:
            # The test in the current node is already in the conjunction, take the correct path
            if prior_tests[(tree.label, tree.value)]:
                return calculate_prob(tree.left, label, value, prior_tests, negate)
            else:
                return calculate_prob(tree.right, label, value, prior_tests, negate)
        elif not (tree.label == label and tree.value == value):
            # The test of current node is not yet in conjunction and is not the test we're looking for
            # Keep propagating (but add weights (estimate how many times the test succeeds/fails))!
            samples_sum = sum(tree.class_probabilities.values())
            left_fraction = sum(tree.left.class_probabilities.values()) / samples_sum
            right_fraction = sum(tree.right.class_probabilities.values()) / samples_sum
            return np.add(left_fraction * calculate_prob(tree.left, label, value, prior_tests, negate),
                          right_fraction * calculate_prob(tree.right, label, value, prior_tests, negate))
        elif not negate:
            # We found the test we are looking for
            # If negate is False, then it is a positive test and we take the left subtree
            return calculate_prob(tree.left, label, value, prior_tests, negate)
        else:
            return calculate_prob(tree.right, label, value, prior_tests, negate)


def calculate_prob_dict(tree, label, value, prior_tests, negate=False):
    """
    Wrapper around calculate_prob, so we know which probability belongs to which class
    """
    return dict(zip(tree.class_probabilities.keys(), calculate_prob(tree, label, value, prior_tests, negate)))


def convert_to_tree(classifier, features):
    """
    Converts the DecisionTreeClassifier from sklearn (adapted CART) to DecisionTree from decisiontree.py

    :param classifier: the trained classifier
    :param features: the features used in the classifier
    :return: a DecisionTree from decisiontree.py
    """
    n_nodes = classifier.tree_.node_count
    children_left = classifier.tree_.children_left
    children_right = classifier.tree_.children_right
    feature = classifier.tree_.feature
    threshold = classifier.tree_.threshold
    classes = classifier.classes_

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes)
    decision_trees = [None] * n_nodes
    for i in range(n_nodes):
        decision_trees[i] = DecisionTree()
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

    for i in range(n_nodes):
        if children_left[i] > 0:
            decision_trees[i].left = decision_trees[children_left[i]]

        if children_right[i] > 0:
            decision_trees[i].right = decision_trees[children_right[i]]

        if is_leaves[i]:
            decision_trees[i].label = classes[np.argmax(classifier.tree_.value[i][0])]
            decision_trees[i].value = None
        else:
            decision_trees[i].label = features[feature[i]]
            decision_trees[i].value = threshold[i]
    return decision_trees[0]


def bootstrap(data, class_label, clf, nr_classifiers=3):
    """
    Bootstrapping ensemble technique

    :param data: a pandas dataframe containing all the data to be bootstrapped
    :param class_label: the column in the dataframe that contains the target variables
    :param clf: sklearn DecisionTreeClassifier
    :param nr_classifiers: the number of required classifiers in the ensemble
    :return: a  vector of fitted classifiers, converted to DecisionTree (decisiontree.py)
    """
    idx = np.random.randint(0, len(data), (nr_classifiers, len(data)))
    decision_trees = []
    for indices in idx:
        X_bootstrap = data.iloc[indices, :].drop(class_label, axis=1).reset_index(drop=True)
        y_bootstrap = data.iloc[indices][class_label].reset_index(drop=True)
        dt = convert_to_tree(clf.fit(X_bootstrap, y_bootstrap), X_bootstrap.columns)
        dt.data = data.iloc[indices, :].reset_index(drop=True)
        dt.populate_samples(X_bootstrap, y_bootstrap)
        decision_trees.append(dt)
    return decision_trees


def ism(decision_trees, data, class_label, min_nr_samples=1, calc_fracs_from_ensemble=False):
    """
    Return a single decision tree from an ensemble of decision tree, using the normalized information gain as
        split criterion, estimated from the ensemble

    :param decision_trees: the ensemble of decision trees, must be of class DecisionTree from decisiontree.py
    :param data: the training data, with labels in column class_label
    :param class_label: the column in data containing the classes
    :param min_nr_samples: pre-prune condition, stop searching if number of samples is smaller or equal than threshold

    :return: a single decision tree based on the ensemble of the different decision trees
    """
    X = data.drop(class_label, axis=1).reset_index(drop=True)
    y = data[class_label].reset_index(drop=True)

    prior_entropy = 0
    tests = set()
    for dt in decision_trees:
        tests = tests.union(extract_tests(dt))
        prior_entropy += calculate_entropy(np.divide(dt.class_probabilities.values(),
                                                     sum(dt.class_probabilities.values())))
    prior_entropy /= len(decision_trees)

    combined_dt = build_dt_from_ensemble(decision_trees, data, class_label, tests, prior_entropy, {}, min_nr_samples)
    combined_dt.populate_samples(X, y)

    return combined_dt


def add_reduce_by_key(A, B):
    """
    Reduces two dicts by key using add operator

    :param A: dict one
    :param B: dict two
    :return: a new dict, containing a of the values if the two dicts have the same key, else just the value
    """
    return {x: A.get(x, 0) + B.get(x, 0) for x in set(A).union(B)}


def build_dt_from_ensemble(decision_trees, data, class_label, tests, prior_entropy, prior_tests={}, min_nr_samples=1,
                           calc_fracs_from_ensemble=False):
    """
    Given an ensemble of decision trees, build a single decision tree using estimates from the ensemble

    :param decision_trees: the ensembles of decision trees
    :param data: the training data frame
    :param class_label: the column with
    :param tests: all possible tests (calculated from the ensemble)
    :param prior_entropy: recursive parameter to calculate information gain
    :param prior_tests: the tests that are already picked for our final decision tree
    :param min_nr_samples: pre-prune condition, the data must be larger than this parameter
    :return: a single decision tree, calculated using information from the ensemble
    """
    # Pre-pruning conditions:
    #   - if the length of data is <= min_nr_samples
    #   - when we have no tests left
    #   - when there is only 1 unique class in the data left
    if len(data) > min_nr_samples and len(tests) > 0 and len(np.unique(data[class_label].values)) > 1:
        max_ig = 0
        best_pos_data, best_neg_data, best_pos_entropy, best_neg_entropy = [None]*4
        best_dt = DecisionTree()
        # Find the test that results in the maximum information gain
        for test in tests:
            pos_avg_probs, neg_avg_probs, pos_fraction, neg_fraction = {}, {}, 0.0, 0.0
            for dt in decision_trees:
                pos_avg_probs = add_reduce_by_key(pos_avg_probs, calculate_prob_dict(dt, test[0], test[1], prior_tests, False))
                neg_avg_probs = add_reduce_by_key(neg_avg_probs, calculate_prob_dict(dt, test[0], test[1], prior_tests, True))
                if calc_fracs_from_ensemble and len(data) > 0:
                    pos_fraction += float(len(dt.data[dt.data[test[0]] <= test[1]]))/len(dt.data)
                    neg_fraction += float(len(dt.data[dt.data[test[0]] > test[1]]))/len(dt.data)

            if calc_fracs_from_ensemble:
                pos_fraction /= float(len(decision_trees))
                neg_fraction /= float(len(decision_trees))

            pos_entropy = calculate_entropy(np.divide(pos_avg_probs.values(), len(decision_trees)))
            neg_entropy = calculate_entropy(np.divide(neg_avg_probs.values(), len(decision_trees)))
            pos_data = data[data[test[0]] <= test[1]].copy()
            neg_data = data[data[test[0]] > test[1]].copy()

            if not calc_fracs_from_ensemble:
                pos_fraction = float(len(pos_data)) / float(len(data))
                neg_fraction = float(len(neg_data)) / float(len(data))

            weighted_entropy = pos_fraction * pos_entropy + neg_fraction * neg_entropy
            information_gain = prior_entropy - weighted_entropy

            if information_gain > max_ig and len(pos_data) > 0 and len(neg_data) > 0:
                max_ig, best_dt.label, best_dt.value = information_gain, test[0], test[1]
                best_pos_data, best_neg_data, best_pos_entropy, best_neg_entropy = pos_data, neg_data, pos_entropy, neg_entropy

        if max_ig == 0:  # If we can't find a test that results in an information gain, we can pre-prune
            return DecisionTree(value=None, label=get_most_occurring_class(data, class_label))

        # Update some variables and do recursive calls
        left_prior_tests = prior_tests.copy()
        left_prior_tests.update({(best_dt.label, best_dt.value): True})
        new_tests = tests.copy()
        new_tests.remove((best_dt.label, best_dt.value))
        best_dt.left = build_dt_from_ensemble(decision_trees, best_pos_data, class_label, new_tests,
                                              best_pos_entropy, left_prior_tests, min_nr_samples)

        right_prior_tests = prior_tests.copy()
        right_prior_tests.update({(best_dt.label, best_dt.value): False})
        best_dt.right = build_dt_from_ensemble(decision_trees, best_neg_data, class_label, new_tests,
                                               best_neg_entropy, right_prior_tests, min_nr_samples)

        return best_dt
    else:
        return DecisionTree(value=None, label=get_most_occurring_class(data, class_label))


# columns = ['Class', 'Alcohol', 'Acid', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols', 'Flavanoids', 'Nonflavanoids',
#           'Proanthocyanins', 'Color', 'Hue', 'Diluted', 'Proline']
# features = ['Alcohol', 'Acid', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols', 'Flavanoids', 'Nonflavanoids',
#           'Proanthocyanins', 'Color', 'Hue', 'Diluted', 'Proline']
# df = pd.read_csv('data/wine.data')
# df.columns = columns
# df['Class'] = np.subtract(df['Class'], 1)

columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Class']
features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
df = pd.read_csv('data/car.data')
df.columns = columns
df = df.reindex(np.random.permutation(df.index)).reset_index(drop=1)

mapping_buy_maint = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
mapping_doors = {'2': 0, '3': 1, '4': 2, '5more': 3}
mapping_persons = {'2': 0, '4': 1, 'more': 2}
mapping_lug = {'small': 0, 'med': 1, 'big': 2}
mapping_safety = {'low': 0, 'med': 1, 'high': 2}
mapping_class = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}

df['maint'] = df['maint'].map(mapping_buy_maint)
df['buying'] = df['buying'].map(mapping_buy_maint)
df['doors'] = df['doors'].map(mapping_doors)
df['persons'] = df['persons'].map(mapping_persons)
df['lug_boot'] = df['lug_boot'].map(mapping_lug)
df['safety'] = df['safety'].map(mapping_safety)
df['Class'] = df['Class'].map(mapping_class).astype(int)


def count_nodes(tree):
    if tree.value is None:
        return 1
    else:
        return count_nodes(tree.left) + count_nodes(tree.right) + 1

N_FOLDS = 5

kf = StratifiedKFold(df['Class'], n_folds=N_FOLDS, shuffle=True, random_state=1337)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_leaf=1, random_state=1337)
rf = RandomForestClassifier(n_estimators=100, random_state=1337)

np.random.seed(1337)

cart_confusion_matrices = []
cart_nodes = []
rf_confusion_matrices = []
ism_confusion_matrices = []
ism_nodes = []

for fold, (train, test) in enumerate(kf):
    print 'Fold', fold
    train = df.iloc[train, :].reset_index(drop=True)
    X_train = train.drop('Class', axis=1).reset_index(drop=True)
    y_train = train['Class'].reset_index(drop=True)
    X_test = df.iloc[test, :].drop('Class', axis=1).reset_index(drop=True)
    y_test = df.iloc[test, :]['Class'].reset_index(drop=True)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cart = convert_to_tree(clf, features)
    cart.populate_samples(X_train, y_train)
    cart_nodes.append(count_nodes(cart))
    y_pred = cart.evaluate_multiple(X_test)
    cart_confusion_matrix = confusion_matrix(y_test.values, y_pred)
    cart_confusion_matrices.append(np.around(np.divide(cart_confusion_matrix, float(np.sum(cart_confusion_matrix))), 4))
    # cart_confusion_matrices.append(np.around(cart_confusion_matrix.astype('float') / cart_confusion_matrix.sum(axis=1)[:, np.newaxis], 4))
    print 'Accuracy CART:', accuracy_score(y_test, y_pred, normalize=1)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_confusion_matrix = confusion_matrix(y_test, y_pred)
    rf_confusion_matrices.append(np.around(np.divide(rf_confusion_matrix, float(np.sum(rf_confusion_matrix))), 4))
    # rf_confusion_matrices.append(np.around(rf_confusion_matrix.astype('float') / rf_confusion_matrix.sum(axis=1)[:, np.newaxis], 4))
    print 'Accuracy RF:', accuracy_score(y_test, y_pred, normalize=1)

    bootstrap_dts = bootstrap(train, 'Class', clf, nr_classifiers=10)
    ism_dt = ism(bootstrap_dts, train, 'Class', min_nr_samples=1, calc_fracs_from_ensemble=True)
    ism_nodes.append(count_nodes(ism_dt))
    y_pred = ism_dt.evaluate_multiple(X_test)
    ism_confusion_matrix = confusion_matrix(y_test, y_pred)
    ism_confusion_matrices.append(np.around(np.divide(ism_confusion_matrix, float(np.sum(ism_confusion_matrix))), 4))
    # ism_confusion_matrices.append(np.around(ism_confusion_matrix.astype('float') / ism_confusion_matrix.sum(axis=1)[:, np.newaxis], 4))
    print 'Accuracy ISM:', accuracy_score(y_test, y_pred, normalize=1)

cart_confusion_matrix = np.mean(cart_confusion_matrices, axis=0)
rf_confusion_matrix = np.mean(rf_confusion_matrices, axis=0)
ism_confusion_matrix = np.mean(ism_confusion_matrices, axis=0)

confusion_matrices = {'CART (' + str(np.mean(cart_nodes)) + ')': cart_confusion_matrix,
                      'Random Forest': rf_confusion_matrix,
                      'ISM (' + str(np.mean(ism_nodes)) + ')': ism_confusion_matrix}
fig = plt.figure()
fig.suptitle('Accuracy on CARS dataset using ' + str(N_FOLDS) + ' folds', fontsize=20)
counter = 0
for key in confusion_matrices:
    ax = fig.add_subplot(1, len(confusion_matrices), counter+1)
    cax = ax.matshow(confusion_matrices[key], cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    diagonal_sum = sum([confusion_matrices[key][i][i] for i in range(len(confusion_matrices[key]))])
    ax.set_title(key + ' (' + str(diagonal_sum) + ')', y=1.08)
    for (j,i),label in np.ndenumerate(confusion_matrices[key]):
        ax.text(i,j,label,ha='center',va='center')
    if counter == len(confusion_matrices)-1:
        fig.colorbar(cax,fraction=0.046, pad=0.04)
    counter += 1

F = plt.gcf()
Size = F.get_size_inches()
F.set_size_inches(Size[0]*2, Size[1], forward=True)
plt.show()