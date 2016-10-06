import time
from bayes_opt import BayesianOptimization
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

from data.load_all_datasets import load_all_datasets
import matplotlib.pyplot as plt


class RFClassification:

    def __init__(self):
        self.clf = None
        self.nr_clf = 0
        self.time = 0

    def construct_rf_classifier(self, train, features, label_col):
        data = train[features]
        target = train[label_col]

        def rfcv(nr_classifiers, max_depth, min_samples_leaf, bootstrap, criterion, max_features):
            nr_classifiers = int(nr_classifiers)
            max_depth = int(max_depth)
            min_samples_leaf = int(min_samples_leaf)
            if np.round(bootstrap):
                bootstrap = True
            else:
                bootstrap = False
            if np.round(criterion):
                criterion = 'gini'
            else:
                criterion = 'entropy'
            if np.round(max_features):
                max_features = None
            else:
                max_features = 1.0

            return cross_val_score(RandomForestClassifier(n_estimators=nr_classifiers, max_depth=max_depth,
                                                          min_samples_leaf=min_samples_leaf, bootstrap=bootstrap,
                                                          criterion=criterion, max_features=max_features),
                                   data, target, 'accuracy', cv=5).mean()

        params = {
            'nr_classifiers': (10, 1000),
            'max_depth': (5, 10),
            'min_samples_leaf': (2, 10),
            'bootstrap': (0, 1),
            'criterion': (0, 1),
            'max_features': (0, 1)
        }

        rfBO = BayesianOptimization(rfcv, params, verbose=0)
        rfBO.maximize(init_points=10, n_iter=20, n_restarts_optimizer=50)

        best_params = rfBO.res['max']['max_params']

        best_nr_classifiers = int(best_params['nr_classifiers'])
        self.nr_clf = best_nr_classifiers
        best_max_depth = int(best_params['max_depth'])
        best_min_samples_leaf = int(best_params['min_samples_leaf'])
        best_bootstrap = best_params['bootstrap']
        best_criterion = best_params['criterion']
        best_max_features = best_params['max_features']

        if np.round(best_bootstrap):
            best_bootstrap = True
        else:
            best_bootstrap = False
        if np.round(best_criterion):
            best_criterion = 'gini'
        else:
            best_criterion = 'entropy'
        if np.round(best_max_features):
            best_max_features = None
        else:
            best_max_features = 1.0

        self.clf = RandomForestClassifier(n_estimators=best_nr_classifiers, max_depth=best_max_depth,
                                          min_samples_leaf=best_min_samples_leaf, bootstrap=best_bootstrap,
                                          criterion=best_criterion, max_features=best_max_features)
        start = time.time()
        self.clf.fit(data, target)
        self.time = time.time() - start

    def evaluate_multiple(self, feature_vectors):
        return self.clf.predict(feature_vectors)

# for dataset in load_all_datasets():
#     df, features, label_col, dataset_name = dataset['dataframe'], dataset['feature_cols'], dataset['label_col'], dataset['name']
#
#     NR_FOLDS = 5
#     skf = StratifiedKFold(df[label_col], n_folds=NR_FOLDS, shuffle=True, random_state=1337)
#
#     accs = []
#     balaccs = []
#     times = []
#     trees = []
#     conf_matrices = {'XGB': []}
#     for fold, (train_idx, test_idx) in enumerate(skf):
#         # print 'Fold', fold+1, '/', NR_FOLDS
#         train = df.iloc[train_idx, :].reset_index(drop=True)
#         X_train = train.drop(label_col, axis=1)
#         y_train = train[label_col]
#         test = df.iloc[test_idx, :].reset_index(drop=True)
#         X_test = test.drop(label_col, axis=1)
#         y_test = test[label_col]
#
#         rf_clf = RFClassification()
#         rf_clf.construct_rf_classifier(train, features, label_col)
#         predictions = rf_clf.evaluate_multiple(X_test)
#
#         conf_matrix = confusion_matrix(y_test, predictions)
#         conf_matrices['XGB'].append(conf_matrix)
#         accs.append(float(sum([conf_matrix[i][i] for i in range(len(conf_matrix))]))/float(np.sum(conf_matrix)))
#         balaccs.append(sum([float(conf_matrix[i][i])/float(sum(conf_matrix[i])) for i in range(len(conf_matrix))])/conf_matrix.shape[0])
#         trees.append(rf_clf.nr_clf)
#         times.append(rf_clf.time)
#     print dataset_name, ':'
#     print 'acc:', np.mean(accs), np.std(accs)
#     print 'balacc:', np.mean(balaccs), np.std(balaccs)
#     print 'times:', np.mean(times), np.std(times)
#     print 'trees:', np.mean(trees), np.std(trees)
#
#     fig = plt.figure()
#     fig.suptitle('Accuracy on ' + dataset['name'] + ' dataset using ' + str(NR_FOLDS) + ' folds', fontsize=20)
#     counter = 0
#     conf_matrices_mean = {}
#     for key in conf_matrices:
#         conf_matrices_mean[key] = np.zeros(conf_matrices[key][0].shape)
#         for i in range(len(conf_matrices[key])):
#             conf_matrices_mean[key] = np.add(conf_matrices_mean[key], conf_matrices[key][i])
#         cm_normalized = np.around(
#             conf_matrices_mean[key].astype('float') / conf_matrices_mean[key].sum(axis=1)[:,
#                                                       np.newaxis], 4)
#
#         diagonal_sum = sum(
#             [conf_matrices_mean[key][i][i] for i in range(len(conf_matrices_mean[key]))])
#         norm_diagonal_sum = sum(
#             [conf_matrices_mean[key][i][i] / sum(conf_matrices_mean[key][i]) for i in
#              range(len(conf_matrices_mean[key]))])
#         total_count = np.sum(conf_matrices_mean[key])
#         print conf_matrices_mean[key], float(diagonal_sum) / float(total_count)
#         print 'Balanced accuracy: ', float(norm_diagonal_sum) / conf_matrices_mean[key].shape[0]
#
#         ax = fig.add_subplot(2, np.math.ceil(len(conf_matrices) / 2.0), counter + 1)
#         cax = ax.matshow(cm_normalized, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
#         ax.set_title(key, y=1.08)
#         for (j, i), label in np.ndenumerate(cm_normalized):
#             ax.text(i, j, label, ha='center', va='center')
#         if counter == len(conf_matrices) - 1:
#             fig.colorbar(cax, fraction=0.046, pad=0.04)
#         counter += 1
#
#     F = plt.gcf()
#     Size = F.get_size_inches()
#     F.set_size_inches(Size[0] * 2, Size[1] * 1.75, forward=True)
#     # plt.show()
#     plt.savefig('../output/' + dataset['name'] + '_CV' + str(NR_FOLDS) + 'RF.png', bbox_inches='tight')