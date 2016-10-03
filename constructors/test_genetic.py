import time
from imblearn.over_sampling import SMOTE
from pandas import DataFrame, Series
import numpy as np
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import operator

from ISM_v3 import bootstrap
from ISM_v3 import ism
from constructors.c45orangeconstructor import C45Constructor
from constructors.cartconstructor import CARTConstructor
from constructors.cn2rulelearner import CN2UnorderedConstructor
from constructors.guideconstructor import GUIDEConstructor
from constructors.questbenchconstructor import QUESTBenchConstructor
from constructors.questconstructor import QuestConstructor
from constructors.treemerger_clean import DecisionTreeMergerClean
from constructors.xgboostconstructor import XGBClassifiction
from data.load_all_datasets import load_all_datasets

import matplotlib.pyplot as plt
import pylab as pl

from inTrees import inTreesClassifier
from write_latex import write_figures
from write_latex import write_footing
from write_latex import write_measurements
from write_latex import write_preamble


def get_best_c45_classifier(train, label_col, skf_tune):
    c45 = C45Constructor()
    cfs = np.arange(0.05, 1.05, 0.05)
    cfs_errors = {}
    for cf in cfs:  cfs_errors[cf] = []

    for train_tune_idx, val_tune_idx in skf_tune:
        train_tune = train.iloc[train_tune_idx, :]
        X_train_tune = train_tune.drop(label_col, axis=1)
        y_train_tune = train_tune[label_col]
        val_tune = train.iloc[val_tune_idx, :]
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


def get_best_cart_classifier(train, label_col, skf_tune):
    cart = CARTConstructor()
    max_depths = np.arange(1,21,2)
    max_depths = np.append(max_depths, None)
    min_samples_splits = np.arange(1,20,1)

    errors = {}
    for max_depth in max_depths:
        for min_samples_split in min_samples_splits:
            errors[(max_depth, min_samples_split)] = []

    for train_tune_idx, val_tune_idx in skf_tune:
        train_tune = train.iloc[train_tune_idx, :]
        X_train_tune = train_tune.drop(label_col, axis=1)
        y_train_tune = train_tune[label_col]
        val_tune = train.iloc[val_tune_idx, :]
        X_val_tune = val_tune.drop(label_col, axis=1)
        y_val_tune = val_tune[label_col]
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                cart.max_depth = max_depth
                cart.min_samples_split = min_samples_split
                tree = cart.construct_tree(X_train_tune, y_train_tune)
                predictions = tree.evaluate_multiple(X_val_tune).astype(int)
                errors[((max_depth, min_samples_split))].append(1 - accuracy_score(predictions, y_val_tune, normalize=True))


    for max_depth in max_depths:
        for min_samples_split in min_samples_splits:
            errors[(max_depth, min_samples_split)] = np.mean(errors[(max_depth, min_samples_split)])

    best_params = min(errors.items(), key=operator.itemgetter(1))[0]
    cart.max_depth = best_params[0]
    cart.min_samples_split = best_params[1]

    return cart

def get_best_cn2_classifier(train, label_col, skf_tune):
    cn2 = CN2UnorderedConstructor()
    beam_widths = np.arange(1,20,3)
    # alphas = np.arange(0.1, 1, 0.2)
    alphas = [0.25, 0.5, 0.75]

    errors = {}
    for beam_width in beam_widths:
        for alpha in alphas:
            errors[(beam_width, alpha)] = []

    for train_tune_idx, val_tune_idx in skf_tune:
        train_tune = train.iloc[train_tune_idx, :]
        X_train_tune = train_tune.drop(label_col, axis=1)
        y_train_tune = train_tune[label_col]
        val_tune = train.iloc[val_tune_idx, :]
        X_val_tune = val_tune.drop(label_col, axis=1)
        y_val_tune = val_tune[label_col]
        for beam_width in beam_widths:
            for alpha in alphas:
                cn2.beam_width = beam_width
                cn2.alpha = alpha
                cn2.extract_rules(X_train_tune, y_train_tune)
                predictions = map(int, [prediction[0].value for prediction in cn2.classify(X_val_tune)])
                errors[(beam_width, alpha)].append(1 - accuracy_score(predictions, y_val_tune, normalize=True))
                # print 1 - accuracy_score(predictions, y_val_tune, normalize=True), (beam_width, alpha)

    for beam_width in beam_widths:
        for alpha in alphas:
            errors[(beam_width, alpha)] = np.mean(errors[(beam_width, alpha)])

    best_params = min(errors.items(), key=operator.itemgetter(1))[0]
    cn2.beam_width = best_params[0]
    cn2.alpha = best_params[1]

    return cn2


def get_metrics(confusion_matrices, model_sizes, execution_times):
    # The parameters of this function are dictionaries with shared keys
    metrics = {}
    for algorithm in confusion_matrices:
        metrics[algorithm] = {}
        accuracies = []
        balaccuracies = []
        for i in range(len(confusion_matrices[algorithm])):
            conf_matrix = confusion_matrices[algorithm][i]
            diagonal_sum = sum([conf_matrix[i][i] for i in range(len(conf_matrix))])
            norm_diagonal_sum = sum([float(conf_matrix[i][i]) / float(sum(conf_matrix[i])) for i in range(len(conf_matrix))])
            total_count = np.sum(conf_matrix)
            accuracies.append(float(diagonal_sum) / float(total_count))
            balaccuracies.append(float(norm_diagonal_sum) / conf_matrix.shape[0])

        metrics[algorithm]['acc'] = (np.around([np.mean(accuracies)], 4)[0], np.around([np.std(accuracies)], 2)[0])
        metrics[algorithm]['balacc'] = (np.around([np.mean(balaccuracies)], 4)[0], np.around([np.std(balaccuracies)], 2)[0])
        metrics[algorithm]['nodes'] = (np.around([np.mean(model_sizes[algorithm])], 4)[0], np.around([np.std(model_sizes[algorithm])], 2)[0])
        metrics[algorithm]['time'] = (np.around([np.mean(execution_times[algorithm])], 4)[0], np.around([np.std(execution_times[algorithm])], 2)[0])

    return metrics


def write_to_file(filename, dataset_name, figure, confusion_matrices, model_sizes, execution_times, ALGORITHMS_PER_TABLE=3):
    target = open(filename, 'w')

    measurements = {dataset_name: get_metrics(confusion_matrices, model_sizes, execution_times)}
    print measurements
    algorithms = measurements[measurements.keys()[0]].keys()
    measurements_list = []
    nr_of_tables = int(np.ceil(float(len(algorithms)) / float(ALGORITHMS_PER_TABLE)))

    for i in range(nr_of_tables - 1):
        table_algorithms = algorithms[i * ALGORITHMS_PER_TABLE:(i + 1) * ALGORITHMS_PER_TABLE]
        print table_algorithms
        measurements_temp = {}
        for dataset in measurements.keys():
            print dataset
            measurements_temp[dataset] = {}
            for algorithm in table_algorithms:
                measurements_temp[dataset][algorithm] = measurements[dataset][algorithm]
        measurements_list.append(measurements_temp)

    last_table_algorithms = algorithms[(nr_of_tables - 1) * ALGORITHMS_PER_TABLE:]
    measurements_temp = {}
    for dataset in measurements.keys():
        measurements_temp[dataset] = {}
        for algorithm in last_table_algorithms:
            measurements_temp[dataset][algorithm] = measurements[dataset][algorithm]
    measurements_list.append(measurements_temp)

    target.write(write_preamble())
    for measurements_ in measurements_list:
        target.write(write_measurements(measurements_))
    target.write(write_figures(figure))
    target.write(write_footing())

    target.close()

datasets = load_all_datasets()
quest_bench = QUESTBenchConstructor()
guide = GUIDEConstructor()
quest = QuestConstructor()
inTrees = inTreesClassifier()
merger = DecisionTreeMergerClean()
NR_FOLDS = 3
for dataset in datasets:
    print dataset['name'], len(dataset['dataframe'])
    conf_matrices = {'QUESTGilles': [], 'GUIDE': [], 'C4.5': [], 'CART': [], 'ISM': [], 'ISM_pruned': [],
                     'Genetic': [], 'CN2': [], 'QUESTLoh': [], 'inTrees': [], 'XGBoost': []}  #
    avg_nodes = {'QUESTGilles': [], 'GUIDE': [], 'C4.5': [], 'CART': [], 'ISM': [], 'ISM_pruned': [],
                 'Genetic': [], 'CN2': [], 'QUESTLoh': [], 'inTrees': [], 'XGBoost': []}  #
    times = {'QUESTGilles': [], 'GUIDE': [], 'C4.5': [], 'CART': [], 'ISM': [], 'ISM_pruned': [],
                 'Genetic': [], 'CN2': [], 'QUESTLoh': [], 'inTrees': [], 'XGBoost': []}  #
    df = dataset['dataframe']
    label_col = dataset['label_col']
    feature_cols = dataset['feature_cols']
    skf = StratifiedKFold(df[label_col], n_folds=NR_FOLDS, shuffle=True, random_state=1337)
    # skf = StratifiedShuffleSplit(df[label_col], 1, test_size=0.33, random_state=1337)

    for fold, (train_idx, test_idx) in enumerate(skf):
        # print 'Fold', fold+1, '/', NR_FOLDS
        train = df.iloc[train_idx, :].reset_index(drop=True)
        X_train = train.drop(label_col, axis=1)
        y_train = train[label_col]
        test = df.iloc[test_idx, :].reset_index(drop=True)
        X_test = test.drop(label_col, axis=1)
        y_test = test[label_col]

        # smote = SMOTE(ratio='auto', kind='regular')
        # print len(X_train)
        # X_train, y_train = smote.fit_sample(X_train, y_train)
        # X_train = DataFrame(X_train, columns=feature_cols)
        # y_train = DataFrame(y_train, columns=[label_col])[label_col]
        # perm = np.random.permutation(len(X_train))
        # X_train = X_train.iloc[perm].reset_index(drop=True)
        # y_train = y_train.iloc[perm].reset_index(drop=True)
        # train = X_train.copy()
        # train[y_train.name] = Series(y_train, index=train.index)
        # print len(X_train)
        #

        print 'QUEST Loh'
        start = time.time()
        quest_bench_tree = quest_bench.construct_tree(X_train, y_train)
        end = time.time()
        times['QUESTLoh'].append(end-start)
        predictions = quest_bench_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['QUESTLoh'].append(confusion_matrix(y_test, predictions))
        print conf_matrices['QUESTLoh'][len(conf_matrices['QUESTLoh']) - 1]
        avg_nodes['QUESTLoh'].append(quest_bench_tree.count_nodes())

        print 'QUEST Gilles'
        start = time.time()
        quest_tree = quest.construct_tree(X_train, y_train)
        end = time.time()
        times['QUESTGilles'].append(end-start)
        predictions = quest_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['QUESTGilles'].append(confusion_matrix(y_test, predictions))
        print conf_matrices['QUESTGilles'][len(conf_matrices['QUESTGilles']) - 1]
        avg_nodes['QUESTGilles'].append(quest_tree.count_nodes())

        print 'GUIDE'
        start = time.time()
        guide_tree = guide.construct_tree(X_train, y_train)
        end = time.time()
        times['GUIDE'].append(end-start)
        predictions = guide_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['GUIDE'].append(confusion_matrix(y_test, predictions))
        print conf_matrices['GUIDE'][len(conf_matrices['GUIDE']) - 1]
        avg_nodes['GUIDE'].append(guide_tree.count_nodes())

        skf_tune = StratifiedKFold(train[label_col], n_folds=3, shuffle=True, random_state=1337)

        print 'C4.5'
        c45_clf = get_best_c45_classifier(train, label_col, skf_tune)
        start = time.time()
        c45_tree = c45_clf.construct_tree(X_train, y_train)
        end = time.time()
        times['C4.5'].append(end-start)
        predictions = c45_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['C4.5'].append(confusion_matrix(y_test, predictions))
        print conf_matrices['C4.5'][len(conf_matrices['C4.5']) - 1]
        avg_nodes['C4.5'].append(c45_tree.count_nodes())

        print 'CART'
        cart_clf = get_best_cart_classifier(train, label_col, skf_tune)
        start = time.time()
        cart_tree = cart_clf.construct_tree(X_train, y_train)
        end = time.time()
        times['CART'].append(end-start)
        predictions = cart_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['CART'].append(confusion_matrix(y_test, predictions))
        print conf_matrices['CART'][len(conf_matrices['CART']) - 1]
        avg_nodes['CART'].append(cart_tree.count_nodes())

        print 'CN2'
        cn2_clf = get_best_cn2_classifier(train, label_col, skf_tune)
        start = time.time()
        cn2 = cn2_clf.extract_rules(X_train, y_train)
        end = time.time()
        times['CN2'].append(end-start)
        predictions = map(int, [prediction[0].value for prediction in cn2_clf.classify(X_test)])
        conf_matrices['CN2'].append(confusion_matrix(y_test, predictions))
        print conf_matrices['CN2'][len(conf_matrices['CN2']) - 1]
        avg_nodes['CN2'].append(len(cn2_clf.model.rules))

        print 'XGBoost'
        xgb_clf = XGBClassifiction()
        xgb_clf.construct_xgb_classifier(train, feature_cols, label_col)
        predictions = xgb_clf.evaluate_multiple(X_test)
        conf_matrix = confusion_matrix(y_test, predictions)
        conf_matrices['XGB'].append(conf_matrix)
        print conf_matrix
        avg_nodes[['XGBoost']].append(xgb_clf.nr_clf)
        times['XGBoost'].append(xgb_clf.time)


        print 'Got all trees, lets merge them!'
        trees = [quest_tree, guide_tree, c45_tree, cart_tree] # quest_bench_tree

        constructors = [c45_clf, cart_clf, quest, guide, quest_bench]
        #
        print 'inTrees'
        start = time.time()
        orl = inTrees.construct_rule_list(train, label_col, constructors, nr_bootstraps=15)
        end = time.time()
        times['inTrees'].append(end-start)
        predictions = orl.evaluate_multiple(X_test).astype(int)
        conf_matrices['inTrees'].append(confusion_matrix(y_test, predictions))
        print conf_matrices['inTrees'][len(conf_matrices['inTrees']) - 1]
        avg_nodes['inTrees'].append(len(orl.rule_list))

        start = time.time()
        ism_tree = ism(bootstrap(train, label_col, constructors, boosting=True, nr_classifiers=15), train, label_col,
                       min_nr_samples=1, calc_fracs_from_ensemble=False)
        end = time.time()
        times['ISM'].append(end-start)
        # ism_tree.visualise('ISM')
        predictions = ism_tree.evaluate_multiple(X_test).astype(int)
        conf_matrices['ISM'].append(confusion_matrix(y_test, predictions))
        avg_nodes['ISM'].append(ism_tree.count_nodes())

        print 'Lets prune the tree'
        start = time.time()
        ism_pruned = ism_tree.cost_complexity_pruning(X_train, y_train, 'ism', ism_constructors=constructors,
                                                      ism_calc_fracs=False, n_folds=3, ism_nr_classifiers=15,
                                                      ism_boosting=True)
        end = time.time()
        times['ISM_pruned'].append(end-start)
        predictions = ism_pruned.evaluate_multiple(X_test).astype(int)
        conf_matrices['ISM_pruned'].append(confusion_matrix(y_test, predictions))
        print conf_matrices['ISM'][len(conf_matrices['ISM']) - 1]
        print conf_matrices['ISM_pruned'][len(conf_matrices['ISM_pruned']) - 1]
        avg_nodes['ISM_pruned'].append(ism_pruned.count_nodes())

        train_gen = train.rename(columns={'Class':'cat'})
        start = time.time()

        # genetic = merger.genetic_algorithm(train_gen, 'cat', constructors, seed=1337, num_iterations=8,
        #                                    num_crossovers=10, population_size=150, val_fraction=0.35, prune=True,
        #                                    max_samples=3, tournament_size=10, nr_bootstraps=10)
        #
        genetic = merger.genetic_algorithm(train_gen, 'cat', constructors, seed=1337, num_iterations=15,
                                           num_crossovers=10, population_size=150, val_fraction=0.5, prune=True,
                                           max_samples=1, tournament_size=10, nr_bootstraps=25)

        end = time.time()
        times['Genetic'].append(end-start)
        predictions = genetic.evaluate_multiple(X_test).astype(int)
        conf_matrices['Genetic'].append(confusion_matrix(y_test, predictions))
        print conf_matrices['Genetic'][len(conf_matrices['Genetic']) - 1]
        avg_nodes['Genetic'].append(genetic.count_nodes())


    fig = plt.figure()
    fig.suptitle('Accuracy on ' + dataset['name'] + ' dataset using ' + str(NR_FOLDS) + ' folds', fontsize=20)
    counter = 0
    conf_matrices_mean = {}
    print conf_matrices
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
        print conf_matrices_mean[key], float(diagonal_sum) / float(total_count)
        print 'Balanced accuracy: ', float(norm_diagonal_sum) / conf_matrices_mean[key].shape[0]

        ax = fig.add_subplot(2, np.math.ceil(len(conf_matrices) / 2.0), counter + 1)
        cax = ax.matshow(cm_normalized, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
        ax.set_title(key + '(' + str(sum(avg_nodes[key])/len(avg_nodes[key])) + ')', y=1.08)
        for (j, i), label in np.ndenumerate(cm_normalized):
            ax.text(i, j, label, ha='center', va='center')
        if counter == len(conf_matrices) - 1:
            fig.colorbar(cax, fraction=0.046, pad=0.04)
        counter += 1

    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 2, Size[1] * 1.75, forward=True)
    # plt.show()
    plt.savefig('output/' + dataset['name'] + '_CV'+str(NR_FOLDS)+'genetic3109.png', bbox_inches='tight')

    write_to_file('output/' + dataset['name'] + '_CV'+str(NR_FOLDS)+'genetic3109.tex',  dataset['name'],
                  {dataset['name']: dataset['name'] + '_CV'+str(NR_FOLDS)+'genetic3109.png'}, conf_matrices, avg_nodes, times,
                  ALGORITHMS_PER_TABLE=3)

# def classification_metrics(confusion_matrices):
#     accuracies = []
#     bal_accuracies = []
#     for conf_matrix in confusion_matrices:
#         diagonal_sum = sum([conf_matrix[i][i] for i in range(len(conf_matrix))])
#         print [conf_matrix[i][i]/sum(conf_matrix[i]) for i in range(len(conf_matrix))]
#         norm_diagonal_sum = sum([float(conf_matrix[i][i])/float(sum(conf_matrix[i])) for i in range(len(conf_matrix))])
#         total_count = np.sum(conf_matrix)
#         accuracies.append(float(diagonal_sum) / float(total_count))
#         bal_accuracies.append(float(norm_diagonal_sum) / conf_matrix.shape[0])
#     return {'acc': (np.around([np.mean(accuracies)], 4)[0], np.around([np.std(accuracies)], 2)[0]),
#             'balacc': (np.around([np.mean(bal_accuracies)], 4)[0], np.around([np.std(bal_accuracies)], 2)[0])}








