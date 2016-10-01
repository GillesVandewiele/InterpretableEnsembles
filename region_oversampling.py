import time
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

from constructors.c45orangeconstructor import C45Constructor
from constructors.treemerger import DecisionTreeMerger
from constructors.treemerger_clean import DecisionTreeMergerClean
from data.load_datasets import load_austra
from sklearn import preprocessing

import numpy as np
import pandas as pd
import collections
import operator

merger = DecisionTreeMergerClean()
merger2 = DecisionTreeMerger()
df, features, label, name = load_austra()
c45 = C45Constructor(cf=0.0)

skf = StratifiedShuffleSplit(df[label], 1, test_size=0.25, random_state=1337)

feature_mins = {}
feature_maxs = {}
for feature in features:
    feature_mins[feature] = 0.0
    feature_maxs[feature] = 1.0

for fold, (train_idx, test_idx) in enumerate(skf):
    # print 'Fold', fold+1, '/', NR_FOLDS
    train = df.iloc[train_idx, :].reset_index(drop=True)
    X_train = train.drop(label, axis=1)
    x = X_train.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X_train = pd.DataFrame(x_scaled)
    X_train.columns = features
    y_train = train[label]
    # print X_train

    test = df.iloc[test_idx, :].reset_index(drop=True)
    X_test = test.drop(label, axis=1)
    x = X_test.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X_test = pd.DataFrame(x_scaled)
    X_test.columns = features
    y_test = test[label]

    count_dict = dict(collections.Counter(train['Class']).items())
    # print count_dict
    max_class, max_count = max(count_dict.iteritems(), key=operator.itemgetter(1))
    # print max_class

    tree1 = c45.construct_tree(X_train, y_train)
    tree1.populate_samples(X_train, y_train)
    tree1.visualise('noSampling')
    predictions = tree1.evaluate_multiple(X_test)
    # print predictions
    # print y_test
    print confusion_matrix(y_test.astype(str), predictions.astype(str))

    regions = merger.decision_tree_to_decision_table(tree1, X_train)

    surface_regions = {}
    for _class in np.unique(train[label].values):
        surface_regions[str(_class)] = []

    for region in regions:
        for feature in features:
            if region[feature][0] == float("-inf"):
                region[feature][0] = feature_mins[feature]
            if region[feature][1] == float("inf"):
                region[feature][1] = feature_maxs[feature]

        surface = 1.0
        for feature in features:
            surface *= float(region[feature][1]-region[feature][0])

        surface_regions[max(region['class'].iteritems(), key=operator.itemgetter(1))[0]].append((region, surface))

    surface_regions_exp = {}
    for _class in np.unique(train[label].values):
        surface_regions_exp[str(_class)] = []

    _sum = {}
    for key in surface_regions:
        _sum[key] = 0.0
        for region, surface in surface_regions[key]:
            surface_regions_exp[key].append((region, np.exp(surface)))
            _sum[key] += np.exp(surface)

    surface_regions_softmax = {}
    for _class in np.unique(train[label].values):
        surface_regions_softmax[str(_class)] = []

    for key in surface_regions_exp:
        for region, surface in surface_regions_exp[key]:
            surface_regions_softmax[key].append((region, surface/_sum[key]))

    samples_to_generate = {}
    for _class in count_dict:
        samples_to_generate[_class] = max_count - count_dict[_class]

    print samples_to_generate

    for _class in samples_to_generate:
        new_samples = []
        for region, softmax in surface_regions_softmax[str(_class)]:
            nr_samples = int(np.floor(softmax * samples_to_generate[_class]))
            for i in range(nr_samples):
                new_sample = {}
                new_sample[label] = _class
                for feature in features:
                    new_sample[feature] = region[feature][0] + np.random.rand()*(region[feature][1] - region[feature][0])
                new_samples.append(new_sample)

        print len(new_samples)

    new_df = pd.DataFrame.from_dict(new_samples)

    train = pd.concat([train, new_df]).reset_index(drop=True)

    X_train = train.drop(label, axis=1)
    x = X_train.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X_train = pd.DataFrame(x_scaled)
    X_train.columns = features
    y_train = train[label]

    c45 = C45Constructor(cf=0.3)

    tree2 = c45.construct_tree(X_train, y_train)
    tree2.populate_samples(X_train, y_train)
    tree2.visualise('Sampling')
    predictions = tree2.evaluate_multiple(X_test)
    print predictions
    # print predictions
    # print y_test
    print confusion_matrix(y_test.astype(str), predictions.astype(str))

