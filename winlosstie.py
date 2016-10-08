from scipy.stats import ttest_ind
import glob
import re
import numpy as np
import pandas as pd

#TODO: First, we read in the different .tex files and store the measurements for acc, balacc, times and nodes

# The .tex file consists of 4 tables with each table containing 4 metrics for 3 algorithms
# header \\
# dataset_name & metric1_alg1 & metric2_alg1 & metric3_alg1 & metric4_alg1 & ... \\

def read_file(PATH, DATASET_NAME, NR_FOLDS, CSV_FILE, NR_METRICS):
    # This is the order of the metrics
    accuracies = {}
    bal_accuracies = {}
    model_complexity = {}
    times = {}

    print 'Reading', len(glob.glob(PATH+'/*.tex')), 'tex files'
    print '-'*300

    for file in glob.glob(PATH+'/*.tex'):
        print file
        tex_file = open(file, 'r')
        lines = tex_file.readlines()
        start = 0
        for i, line in enumerate(lines):
            if 'begin{tabular}' in line:
                start = i+1
            if 'end{tabular}' in line:
                data = lines[start:i]
                # First entry of data contains the names of the algorithms, second entry contains the names of the metrics
                # Third contains the measurements, grouped per NR_METRICS
                names = []
                for entry in map(lambda x: x.rstrip().lstrip(), data[0].split('&')):
                    m = re.search('textbf{(.+)}}', entry)
                    if m:
                        algorithm_name = m.group(1)
                        if algorithm_name not in accuracies:
                            accuracies[algorithm_name] = []
                            bal_accuracies[algorithm_name] = []
                            model_complexity[algorithm_name] = []
                            times[algorithm_name] = []

                        names.append(algorithm_name)

                measurement_strings = map(lambda x: x.rstrip().lstrip(), data[2].split('&'))
                for i, name in enumerate(names):
                    measurements_name = measurement_strings[1+i*NR_METRICS:1+(i+1)*NR_METRICS]
                    m = re.search('([0-9\.]+\+[0-9\.]+)\$', measurements_name[0])
                    if m:
                        result = m.group(1).split('+')
                        accuracies[name].append((float(result[0]), float(result[1]), file, DATASET_NAME, NR_FOLDS))
                    m = re.search('([0-9\.]+\+[0-9\.]+)\$', measurements_name[1])
                    if m:
                        result = m.group(1).split('+')
                        bal_accuracies[name].append((float(result[0]), float(result[1]), file, DATASET_NAME, NR_FOLDS))
                    m = re.search('([0-9\.]+\+[0-9\.]+)\$', measurements_name[2])
                    if m:
                        result = m.group(1).split('+')
                        model_complexity[name].append((float(result[0]), float(result[1]), file, DATASET_NAME, NR_FOLDS))
                    m = re.search('([0-9\.]+\+[0-9\.]+)\$', measurements_name[3])
                    if m:
                        result = m.group(1).split('+')
                        times[name].append((float(result[0]), float(result[1]), file, DATASET_NAME, NR_FOLDS))
                    # accuracies[name].append(measurement_strings[1:1+NR_METRICS])
                    # bal_accuracies[name].append(measurement_strings[1+NR_METRICS:1+2*NR_METRICS])
                    # model_complexity[name].append(measurement_strings[1+2*NR_METRICS:1+3*NR_METRICS])
                    # times[name].append(measurement_strings[1+3*NR_METRICS:1+4*NR_METRICS])

    print '==> Did we succeed?', len(accuracies['CN2']) == len(glob.glob(PATH+'/*.tex'))
    print '-'*300

    results_df = pd.read_csv(CSV_FILE)
    cols = results_df.columns.values

    for algorithm in accuracies.keys():
        for entry in accuracies[algorithm]:
                results_df= pd.concat([pd.DataFrame([[entry[2],entry[3],entry[4],algorithm,'accuracy',entry[0],entry[1]]], columns=cols), results_df], ignore_index=True)
        for entry in bal_accuracies[algorithm]:
                results_df= pd.concat([pd.DataFrame([[entry[2],entry[3],entry[4],algorithm,'bal_accuracy',entry[0],entry[1]]], columns=cols), results_df], ignore_index=True)
        for entry in times[algorithm]:
                results_df= pd.concat([pd.DataFrame([[entry[2],entry[3],entry[4],algorithm,'time',entry[0],entry[1]]], columns=cols), results_df], ignore_index=True)
        for entry in model_complexity[algorithm]:
                results_df= pd.concat([pd.DataFrame([[entry[2],entry[3],entry[4],algorithm,'complexity',entry[0],entry[1]]], columns=cols), results_df], ignore_index=True)

    results_df.drop_duplicates(inplace=True)
    results_df = results_df.reset_index(drop=True)

    results_df.to_csv(CSV_FILE, index_label=False)

    print 'nr of measurements now:', len(results_df)


PATH = '/home/gvandewiele/Dropbox/Work Documents/Results/Genetic/3-fold CV/ecoli/tex'
DATASET_NAME = 'ecoli'
NR_FOLDS = 3
CSV_FILE = '/home/gvandewiele/Dropbox/Work Documents/Results/Genetic/results.csv'
NR_METRICS = 4

# read_file(PATH, DATASET_NAME, NR_FOLDS, CSV_FILE, NR_METRICS)

results_df = pd.read_csv(CSV_FILE)
#
accuracies = {}
DATASET = 'ecoli'
for algorithm in np.unique(results_df['algorithm']):
    accuracies[algorithm] = results_df[(results_df['dataset'] == DATASET) & (results_df['algorithm'] == algorithm)
                            & (results_df['metric'] == 'accuracy')]['mean'].values

algorithms = accuracies.keys()
for algorithm1 in range(len(algorithms)):
    for algorithm2 in range(algorithm1, len(algorithms)):
        if algorithm1 != algorithm2 and ttest_ind([x for x in accuracies[algorithms[algorithm1]]], [x for x in accuracies[algorithms[algorithm2]]], equal_var=False).pvalue <= 0.05:
            print algorithms[algorithm1], np.mean([x for x in accuracies[algorithms[algorithm1]]])
            print algorithms[algorithm2], np.mean([x for x in accuracies[algorithms[algorithm2]]])
            print '==================='
#
