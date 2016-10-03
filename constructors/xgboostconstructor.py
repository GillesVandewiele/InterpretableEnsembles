import time
import xgboost as xgb
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

from data.load_all_datasets import load_all_datasets

import matplotlib.pyplot as plt

# data = heart_train[features]
# target = heart_train[label_col]
#
#
# def xgbcv(nr_classifiers, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma, reg_lambda):
#     nr_classifiers = int(nr_classifiers)
#     max_depth = int(max_depth)
#     min_child_weight = int(min_child_weight)
#     return cross_val_score(XGBClassifier(learning_rate=learning_rate, n_estimators=nr_classifiers,
#                                          gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree, nthread=1,
#                                          scale_pos_weight=1, reg_lambda=reg_lambda,
#                                          min_child_weight=min_child_weight, max_depth=max_depth),
#                            data, target, 'accuracy', cv=5).mean()
#
# params = {
#     'nr_classifiers': (50, 1000),
#     'learning_rate': (0.01, 0.3),
#     'max_depth': (5, 10),
#     'min_child_weight': (2, 10),
#     'subsample': (0.7, 0.8),
#     'colsample_bytree' :(0.5, 0.99),
#     'gamma': (1., 0.01),
#     'reg_lambda': (0, 1)
# }
#
# xgbBO = BayesianOptimization(xgbcv, params)
# xgbBO.maximize(init_points=10, n_iter=25, n_restarts_optimizer=100)
# print 'RFC:', xgbBO.res['max']['max_params']


class XGBClassifiction:

    def __init__(self):
        self.clf = None
        self.nr_clf = 0
        self.time = 0

    def construct_xgb_classifier(self, train, features, label_col):
        data = train[features]
        target = train[label_col]

        def xgbcv(nr_classifiers, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma,
                  reg_lambda):
            nr_classifiers = int(nr_classifiers)
            max_depth = int(max_depth)
            min_child_weight = int(min_child_weight)
            return cross_val_score(XGBClassifier(learning_rate=learning_rate, n_estimators=nr_classifiers,
                                                 gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree,
                                                 nthread=1, scale_pos_weight=1, reg_lambda=reg_lambda,
                                                 min_child_weight=min_child_weight, max_depth=max_depth),
                                   data, target, 'accuracy', cv=5).mean()

        params = {
            'nr_classifiers': (50, 1000),
            'learning_rate': (0.01, 0.3),
            'max_depth': (5, 10),
            'min_child_weight': (2, 10),
            'subsample': (0.7, 0.8),
            'colsample_bytree': (0.5, 0.99),
            'gamma': (1., 0.01),
            'reg_lambda': (0, 1)
        }

        xgbBO = BayesianOptimization(xgbcv, params, verbose=0)
        xgbBO.maximize(init_points=10, n_iter=25, n_restarts_optimizer=100)

        best_params = xgbBO.res['max']['max_params']

        best_nr_classifiers = int(best_params['nr_classifiers'])
        self.nr_clf = best_nr_classifiers
        best_max_depth = int(best_params['max_depth'])
        best_min_child_weight = int(best_params['min_child_weight'])
        best_colsample_bytree = best_params['colsample_bytree']
        best_subsample = best_params['subsample']
        best_reg_lambda = best_params['reg_lambda']
        best_learning_rate = best_params['learning_rate']
        best_gamma = best_params['gamma']

        self.clf = XGBClassifier(learning_rate=best_learning_rate, n_estimators=best_nr_classifiers,
                                 gamma=best_gamma, subsample=best_subsample, colsample_bytree=best_colsample_bytree,
                                 nthread=1, scale_pos_weight=1, reg_lambda=best_reg_lambda,
                                 min_child_weight=best_min_child_weight, max_depth=best_max_depth)
        start = time.time()
        self.clf.fit(data, target)
        self.time = time.time() - start

    def evaluate_multiple(self, feature_vectors):
        return self.clf.predict(feature_vectors)

for dataset in load_all_datasets():
    df, features, label_col, dataset_name = dataset['dataframe'], dataset['feature_cols'], dataset['label_col'], dataset['name']

    NR_FOLDS = 5
    skf = StratifiedKFold(df[label_col], n_folds=NR_FOLDS, shuffle=True, random_state=1337)

    accs = []
    balaccs = []
    times = []
    trees = []
    conf_matrices = {'XGB': []}
    for fold, (train_idx, test_idx) in enumerate(skf):
        # print 'Fold', fold+1, '/', NR_FOLDS
        train = df.iloc[train_idx, :].reset_index(drop=True)
        X_train = train.drop(label_col, axis=1)
        y_train = train[label_col]
        test = df.iloc[test_idx, :].reset_index(drop=True)
        X_test = test.drop(label_col, axis=1)
        y_test = test[label_col]

        xgb_clf = XGBClassifiction()
        xgb_clf.construct_xgb_classifier(train, features, label_col)
        predictions = xgb_clf.evaluate_multiple(X_test)

        conf_matrix = confusion_matrix(y_test, predictions)
        conf_matrices['XGB'].append(conf_matrix)
        accs.append(float(sum([conf_matrix[i][i] for i in range(len(conf_matrix))]))/float(np.sum(conf_matrix)))
        balaccs.append(sum([float(conf_matrix[i][i])/float(sum(conf_matrix[i])) for i in range(len(conf_matrix))])/conf_matrix.shape[0])
        trees.append(xgb_clf.nr_clf)
        times.append(xgb_clf.time)
    print dataset_name, ':'
    print 'acc:', np.mean(accs), np.std(accs)
    print 'balacc:', np.mean(balaccs), np.std(balaccs)
    print 'times:', np.mean(times), np.std(times)
    print 'trees:', np.mean(trees), np.std(trees)

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
            [conf_matrices_mean[key][i][i] / sum(conf_matrices_mean[key][i]) for i in
             range(len(conf_matrices_mean[key]))])
        total_count = np.sum(conf_matrices_mean[key])
        print conf_matrices_mean[key], float(diagonal_sum) / float(total_count)
        print 'Balanced accuracy: ', float(norm_diagonal_sum) / conf_matrices_mean[key].shape[0]

        ax = fig.add_subplot(2, np.math.ceil(len(conf_matrices) / 2.0), counter + 1)
        cax = ax.matshow(cm_normalized, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
        ax.set_title(key, y=1.08)
        for (j, i), label in np.ndenumerate(cm_normalized):
            ax.text(i, j, label, ha='center', va='center')
        if counter == len(conf_matrices) - 1:
            fig.colorbar(cax, fraction=0.046, pad=0.04)
        counter += 1

    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 2, Size[1] * 1.75, forward=True)
    # plt.show()
    plt.savefig('../output/' + dataset['name'] + '_CV' + str(NR_FOLDS) + 'XGB.png', bbox_inches='tight')

