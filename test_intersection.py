import time
from sklearn.cross_validation import StratifiedShuffleSplit

from constructors.c45orangeconstructor import C45Constructor
from constructors.treemerger import DecisionTreeMerger
from constructors.treemerger_clean import DecisionTreeMergerClean
from data.load_datasets import load_led7

import numpy as np


merger = DecisionTreeMergerClean()
merger2 = DecisionTreeMerger()
df, features, label, name = load_led7()
c45 = C45Constructor()

skf = StratifiedShuffleSplit(df[label], 1, test_size=0.5, random_state=1337)

feature_mins = {}
feature_maxs = {}
for feature in features:
    feature_mins[feature] = np.min(df[feature])
    feature_maxs[feature] = np.max(df[feature])

for fold, (train_idx, test_idx) in enumerate(skf):
    # print 'Fold', fold+1, '/', NR_FOLDS
    train = df.iloc[train_idx, :].reset_index(drop=True)
    X_train = train.drop(label, axis=1)
    y_train = train[label]
    test = df.iloc[test_idx, :].reset_index(drop=True)
    X_test = test.drop(label, axis=1)
    y_test = test[label]

    tree1 = c45.construct_tree(X_train, y_train)
    tree1.populate_samples(X_train, y_train)
    regions1 = merger.decision_tree_to_decision_table(tree1, X_train)

    # merger2.plot_regions('region1', regions1, np.unique(y_train.values), features[0], features[1],
    #                      x_max=feature_maxs[features[0]], y_max=feature_maxs[features[1]],
    #                      x_min=feature_mins[features[0]], y_min=feature_mins[features[1]])

    tree2 = c45.construct_tree(X_test, y_test)
    tree2.populate_samples(X_test, y_test)
    regions2 = merger.decision_tree_to_decision_table(tree2, X_train)

    # merger2.plot_regions('region2', regions2, np.unique(y_train.values), features[0], features[1],
    #                      x_max=feature_maxs[features[0]], y_max=feature_maxs[features[1]],
    #                      x_min=feature_mins[features[0]], y_min=feature_mins[features[0]])

    start = time.time()
    merged_regions = merger.calculate_intersection(regions1, regions2, features, feature_maxs, feature_mins)
    end = time.time()

    print 'Clean method:', end-start

    start = time.time()
    merged_regions_check = merger2.calculate_intersection(regions1, regions2, features, feature_maxs, feature_mins)
    end = time.time()
    print 'Old method:', end-start

    for region in merged_regions_check:
        print (region in merged_regions)



