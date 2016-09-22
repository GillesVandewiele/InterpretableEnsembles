from sklearn import datasets

import pandas as pd
import numpy as np
import os


# def load_wine():
#     columns = ['Class', 'Alcohol', 'Acid', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols', 'Flavanoids', 'Nonflavanoids',
#               'Proanthocyanins', 'Color', 'Hue', 'Diluted', 'Proline']
#     features = ['Alcohol', 'Acid', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols', 'Flavanoids', 'Nonflavanoids',
#               'Proanthocyanins', 'Color', 'Hue', 'Diluted', 'Proline']
#     df = pd.read_csv(os.path.join(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]), 'wine.data'))
#     df.columns = columns
#     df['Class'] = np.subtract(df['Class'], 1)
#
#     return df, features, 'Class', 'wine'
# #
# #
# def load_cars():
#     columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Class']
#     features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
#     df = pd.read_csv(os.path.join(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]), 'car.data'))
#     df.columns = columns
#     df = df.reindex(np.random.permutation(df.index)).reset_index(drop=1)
#
#     mapping_buy_maint = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
#     mapping_doors = {'2': 0, '3': 1, '4': 2, '5more': 3}
#     mapping_persons = {'2': 0, '4': 1, 'more': 2}
#     mapping_lug = {'small': 0, 'med': 1, 'big': 2}
#     mapping_safety = {'low': 0, 'med': 1, 'high': 2}
#     mapping_class = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
#
#     df['maint'] = df['maint'].map(mapping_buy_maint)
#     df['buying'] = df['buying'].map(mapping_buy_maint)
#     df['doors'] = df['doors'].map(mapping_doors)
#     df['persons'] = df['persons'].map(mapping_persons)
#     df['lug_boot'] = df['lug_boot'].map(mapping_lug)
#     df['safety'] = df['safety'].map(mapping_safety)
#     df['Class'] = df['Class'].map(mapping_class).astype(int)
#
#     return df, features, 'Class', 'cars'
#     # return df.iloc[:150, :], features, 'Class', 'cars'
# #
# #
# def load_wisconsin_breast_cancer():
#     columns = ['ID', 'ClumpThickness', 'CellSizeUniform', 'CellShapeUniform', 'MargAdhesion', 'EpithCellSize', 'BareNuclei',
#                'BlandChromatin', 'NormalNuclei', 'Mitoses', 'Class']
#     features = ['ClumpThickness', 'CellSizeUniform', 'CellShapeUniform', 'MargAdhesion', 'EpithCellSize', 'BareNuclei',
#                'BlandChromatin', 'NormalNuclei', 'Mitoses']
#     df = pd.read_csv(os.path.join(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]), 'breast-cancer-wisconsin.data'))
#     df.columns = columns
#     df['Class'] = np.subtract(np.divide(df['Class'], 2), 1)
#     df = df.drop('ID', axis=1).reset_index(drop=True)
#     df['BareNuclei'] = df['BareNuclei'].replace('?', int(np.mean(df['BareNuclei'][df['BareNuclei'] != '?'].map(int))))
#     df = df.applymap(int)
#
#     return df, features, 'Class', 'wisconsin_breast'
#
#
# def load_heart():
#     columns = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
#                'resting electrocardio', 'max heartrate', 'exercise induced', 'oldpeak', 'slope peak', \
#                'number of vessels', 'thal', 'Class']
#     features = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', \
#                'resting electrocardio', 'max heartrate', 'exercise induced', 'oldpeak', 'slope peak', \
#                'number of vessels', 'thal']
#     df = pd.read_csv(os.path.join(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]), 'heart.dat'), sep=' ')
#     df.columns = columns
#     df['Class'] = np.subtract(df['Class'], 1)
#
#     return df, features, 'Class', 'heart'
#
#
# def load_iris():
#     iris = datasets.load_iris()
#     df = pd.DataFrame(iris.data)
#     features = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
#     df.columns = features
#     df["Class"] = iris.target
#     # df = df.drop('SepalWidth', axis=1)
#     # df = df.drop('PetalLength', axis=1)
#     return df, features, 'Class', 'iris'


def load_shuttle():
    columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8',
               'feature9', 'Class']
    features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8',
               'feature9']

    df = pd.read_csv(os.path.join(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]), 'shuttle.tst'), sep=' ')
    df.columns = columns
    for feature in features:
        if np.min(df[feature]) < 0:
            df[feature] += np.min(df[feature]) * (-1)
    df = df[df['Class'] < 6]
    df['Class'] = np.subtract(df['Class'], 1)
    df = df.reset_index(drop=True)

    return df, features, 'Class', 'shuttle'


def load_nursery():
    columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'Class']
    features = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health']

    mapping_parents = {'usual': 0, 'pretentious': 1, 'great_pret': 2}
    mapping_has_nurs = {'proper': 0, 'less_proper': 1, 'improper': 2, 'critical': 3, 'very_crit': 4}
    mapping_form = {'complete': 0, 'completed': 1, 'incomplete': 2, 'foster': 3}
    mapping_housing = {'convenient': 0, 'less_conv': 1, 'critical': 2}
    mapping_finance = {'convenient': 0, 'inconv': 1}
    mapping_social = {'nonprob': 0, 'slightly_prob': 1, 'problematic': 2}
    mapping_health = {'recommended': 0, 'priority': 1, 'not_recom': 2}
    mapping_class = {'not_recom': 1, 'recommend': 0, 'very_recom': 2, 'priority': 3, 'spec_prior': 4}

    df = pd.read_csv(os.path.join(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]), 'nursery.data'), sep=',')
    df = df.dropna()
    df.columns = columns

    df['parents'] = df['parents'].map(mapping_parents)
    df['has_nurs'] = df['has_nurs'].map(mapping_has_nurs)
    df['form'] = df['form'].map(mapping_form)
    df['children'] = df['children'].map(lambda x: 4 if x == 'more' else int(x))
    df['housing'] = df['housing'].map(mapping_housing)
    df['finance'] = df['finance'].map(mapping_finance)
    df['social'] = df['social'].map(mapping_social)
    df['health'] = df['health'].map(mapping_health)
    df['Class'] = df['Class'].map(mapping_class)

    df = df[df['Class'] != 0]
    df['Class'] = np.subtract(df['Class'], 1)
    df = df.reset_index(drop=True)

    return df, features, 'Class', 'nursery'
