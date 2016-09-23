import time
import operator

from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import TomekLinks

from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import constructors.c45orangeconstructor
from data.load_all_datasets import load_all_datasets
from write_latex import write_figures
from write_latex import write_footing
from write_latex import write_measurements
from write_latex import write_preamble


def get_best_c45_classifier(train, label_col, skf_tune):
    c45 = constructors.c45orangeconstructor.C45Constructor()
    cfs = np.arange(0.05, 1.05, 0.05)
    cfs_errors = {}
    for cf in cfs:  cfs_errors[cf] = []

    for train_tune_idx, val_tune_idx in skf_tune:
        train_tune = train.iloc[train_tune_idx, :].reset_index(drop=True)
        X_train_tune = train_tune.drop(label_col, axis=1)
        y_train_tune = train_tune[label_col]
        val_tune = train.iloc[val_tune_idx, :].reset_index(drop=True)
        X_val_tune = val_tune.drop(label_col, axis=1)
        y_val_tune = val_tune[label_col]
        for cf in cfs:
            c45.cf = cf
            tree = c45.construct_tree(X_train_tune, y_train_tune)
            predictions = tree.evaluate_multiple(X_val_tune).astype(int)
            cfs_errors[cf].append(1 - accuracy_score(predictions, y_val_tune, normalize=True))

    for cf in cfs:
        cfs_errors[cf] = np.mean(cfs_errors[cf])

    c45.cf = min(cfs_errors.items(), key=operator.itemgetter(1))[0]
    return c45


def sample_test(method, X_train, y_train):
    start = time.time()
    prior_nodes = len(X_train)
    X_train_sampled, y_train_sampled = method.fit_sample(X_train, y_train)
    end = time.time()
    X_train_sampled = DataFrame(X_train_sampled, columns=feature_cols)
    y_train_sampled = DataFrame(y_train_sampled, columns=[label_col])[label_col]
    perm = np.random.permutation(len(X_train_sampled))
    X_train_sampled = X_train_sampled.iloc[perm].reset_index(drop=True)
    y_train_sampled = y_train_sampled.iloc[perm].reset_index(drop=True)
    train = X_train_sampled.copy()
    train[y_train_sampled.name] = Series(y_train_sampled, index=train.index)
    print 'From', prior_nodes, 'to', len(X_train_sampled), 'in', (end - start), 'seconds'

    skf_tune = StratifiedKFold(train[label_col], n_folds=3, shuffle=True, random_state=1337)

    c45_clf = get_best_c45_classifier(train, label_col, skf_tune)
    c45_tree = c45_clf.construct_tree(X_train_sampled, y_train_sampled)
    predictions = c45_tree.evaluate_multiple(X_test).astype(int)
    return confusion_matrix(y_test, predictions), c45_tree.count_nodes(), end-start


def classification_metrics(confusion_matrices):
    accuracies = []
    bal_accuracies = []
    for conf_matrix in confusion_matrices:
        diagonal_sum = sum([conf_matrix[i][i] for i in range(len(conf_matrix))])
        print [conf_matrix[i][i]/sum(conf_matrix[i]) for i in range(len(conf_matrix))]
        norm_diagonal_sum = sum([float(conf_matrix[i][i])/float(sum(conf_matrix[i])) for i in range(len(conf_matrix))])
        total_count = np.sum(conf_matrix)
        accuracies.append(float(diagonal_sum) / float(total_count))
        bal_accuracies.append(float(norm_diagonal_sum) / conf_matrix.shape[0])
    return {'acc': (np.around([np.mean(accuracies)], 4)[0], np.around([np.std(accuracies)], 2)[0]),
            'balacc': (np.around([np.mean(bal_accuracies)], 4)[0], np.around([np.std(bal_accuracies)], 2)[0])}


datasets = load_all_datasets()
NR_FOLDS = 5
measurements = {}
figures = {}
for dataset in datasets:
    measurements[dataset['name']] = {}
    print dataset['name']
    conf_matrices = {'Imbalanced': [], 'RUS': [], 'Tomek': [], 'Cluster': [], 'INN': [], 'ENN': [],
                     'ROS': [], 'SMOTE': [], 'SMOTE SVM': [], 'SMOTE Tomek': [], 'SMOTE ENN': []}
    avg_nodes = {'Imbalanced': [], 'RUS': [], 'Tomek': [], 'Cluster': [], 'INN': [], 'ENN': [],
                 'ROS': [], 'SMOTE': [], 'SMOTE SVM': [], 'SMOTE Tomek': [], 'SMOTE ENN': []}
    avg_time = {'Imbalanced': [], 'RUS': [], 'Tomek': [], 'Cluster': [], 'INN': [], 'ENN': [],
                'ROS': [], 'SMOTE': [], 'SMOTE SVM': [], 'SMOTE Tomek': [], 'SMOTE ENN': []}
    methods = {'RUS': RandomUnderSampler(), 'Tomek': TomekLinks(), 'Cluster': ClusterCentroids(),
               'INN': InstanceHardnessThreshold(), 'ENN': RepeatedEditedNearestNeighbours(),
               'ROS': RandomOverSampler(ratio='auto'), 'SMOTE': SMOTE(ratio='auto', kind='regular'),
               'SMOTE SVM': SMOTE(ratio='auto', kind='svm', **{'class_weight': 'balanced'}),
               'SMOTE Tomek': SMOTETomek(ratio='auto'), 'SMOTE ENN': SMOTEENN(ratio='auto')}
    df = dataset['dataframe']
    label_col = dataset['label_col']
    feature_cols = dataset['feature_cols']
    skf = StratifiedKFold(df[label_col], n_folds=NR_FOLDS, shuffle=True, random_state=1337)

    for fold, (train_idx, test_idx) in enumerate(skf):
        print 'Fold', fold+1, '/', NR_FOLDS
        train = df.iloc[train_idx, :].reset_index(drop=True)
        X_train = train.drop(label_col, axis=1)
        y_train = train[label_col]
        test = df.iloc[test_idx, :].reset_index(drop=True)
        X_test = test.drop(label_col, axis=1)
        y_test = test[label_col]

        skf_tune = StratifiedKFold(train[label_col], n_folds=3, shuffle=True, random_state=1337)

        c45_clf = get_best_c45_classifier(train, label_col, skf_tune)
        c45_tree = c45_clf.construct_tree(X_train, y_train)
        predictions = c45_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['Imbalanced'].append(confusion_matrix(y_test, predictions))
        avg_nodes['Imbalanced'].append(c45_tree.count_nodes())
        avg_time['Imbalanced'].append(0)

        for key in methods.keys():
            print key
            confmat, nr_nodes, time_elapsed = sample_test(methods[key], X_train, y_train)
            conf_matrices[key].append(confmat)
            avg_nodes[key].append(nr_nodes)
            avg_time[key].append(time_elapsed)

    fig = plt.figure()
    fig.suptitle('Accuracy on ' + dataset['name'] + ' dataset using ' + str(NR_FOLDS) + ' folds', fontsize=20)
    counter = 0
    conf_matrices_mean = {}
    for key in conf_matrices:
        conf_matrices_mean[key] = np.zeros(conf_matrices[key][0].shape)
        for i in range(len(conf_matrices[key])):
            conf_matrices_mean[key] = np.add(conf_matrices_mean[key], conf_matrices[key][i])
        cm_normalized = np.around(
            conf_matrices_mean[key].astype('float') / conf_matrices_mean[key].sum(axis=1)[:,
                                                      np.newaxis], 4)

        diagonal_sum = sum(
            [conf_matrices_mean[key][i][i] for i in range(len(conf_matrices_mean[key]))])
        norm_diagonal_sum = sum(
            [conf_matrices_mean[key][i][i]/sum(conf_matrices_mean[key][i]) for i in range(len(conf_matrices_mean[key]))])
        total_count = np.sum(conf_matrices_mean[key])
        print key
        print conf_matrices_mean[key], float(diagonal_sum) / float(total_count)
        print 'Balanced accuracy: ', float(norm_diagonal_sum) / conf_matrices_mean[key].shape[0]
        print np.mean(avg_time[key])
        print np.mean(avg_nodes[key])

        ax = fig.add_subplot(2, np.math.ceil(len(conf_matrices) / 2.0), counter + 1)
        cax = ax.matshow(cm_normalized, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
        ax.set_title(key + '(' + str(sum(avg_nodes[key])/len(avg_nodes[key])) + ')', y=1.12)
        for (j, i), label in np.ndenumerate(cm_normalized):
            ax.text(i, j, label, ha='center', va='center')
        if counter == len(conf_matrices) - 1:
            fig.colorbar(cax, fraction=0.046, pad=0.04)
        counter += 1

        measurements[dataset['name']][key] = classification_metrics(conf_matrices[key])
        measurements[dataset['name']][key]['nodes'] = (np.around([np.mean(avg_nodes[key])], 2)[0],
                                                        np.around([np.std(avg_nodes[key])], 2)[0])
        measurements[dataset['name']][key]['time'] = (np.around([np.mean(avg_time[key])], 2)[0],
                                                      np.around([np.std(avg_time[key])], 2)[0])

    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * (np.ceil(conf_matrices_mean[conf_matrices.keys()[0]].shape[0] / 2.0)+0.5), Size[1] * 2, forward=True)
    plt.savefig('output/'+dataset['name']+'_balancing.png', bbox_inches='tight')
    figures[dataset['name']] = 'output/'+dataset['name']+'_balancing.png'


print figures
algorithms = measurements[measurements.keys()[0]].keys()
algorithms_half1 = algorithms[:len(algorithms)/2]
algorithms_half2 = algorithms[len(algorithms)/2:]
measurements1 = {}
measurements2 = {}
datasets = measurements.keys()
for dataset in datasets:
    measurements1[dataset] = {}
    measurements2[dataset] = {}
    for algorithm in algorithms_half1:
        measurements1[dataset][algorithm] = measurements[dataset][algorithm]
    for algorithm in algorithms_half2:
        measurements2[dataset][algorithm] = measurements[dataset][algorithm]
print write_preamble()
print write_measurements(measurements1)
print write_measurements(measurements2)
print write_figures(figures)
print write_footing()


