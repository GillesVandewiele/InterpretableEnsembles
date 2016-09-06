import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from constructors.cartconstructor import CARTConstructor

# columns = ['ID', 'ClumpThickness', 'CellSizeUniform', 'CellShapeUniform', 'MargAdhesion', 'EpithCellSize', 'BareNuclei',
#            'BlandChromatin', 'NormalNuclei', 'Mitoses', 'Class']
# features = ['ClumpThickness', 'CellSizeUniform', 'CellShapeUniform', 'MargAdhesion', 'EpithCellSize', 'BareNuclei',
#            'BlandChromatin', 'NormalNuclei', 'Mitoses']
# df = pd.read_csv('data/breast-cancer-wisconsin.data')
# df.columns = columns
# df['Class'] = np.subtract(np.divide(df['Class'], 2), 1)
# df = df.drop('ID', axis=1).reset_index(drop=True)
# df['BareNuclei'] = df['BareNuclei'].replace('?', int(np.mean(df['BareNuclei'][df['BareNuclei'] != '?'].map(int))))
# df = df.applymap(int)
#
#
# X = df.drop('Class', axis=1).reset_index(drop=True)
# y = df['Class'].reset_index(drop=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1337)

data = [[1, 1, 0], [2, 1, 0], [3, 1, 0], [4, 1, 0],
        [1, 2, 1], [2, 2, 1], [3, 2, 1], [4, 2, 1],
        [1, 3, 1], [2, 3, 1], [3, 3, 0], [4, 3, 0],
        [1, 4, 1], [2, 4, 1], [3, 4, 0], [4, 4, 0]]
X = pd.DataFrame([sample[:2] for sample in data])
X.columns = ['X', 'Y']
y = pd.DataFrame([sample[2] for sample in data])
y.columns = ['Class']

cart = CARTConstructor(max_depth=None)
dt = cart.construct_tree(X, y)
dt.populate_samples(X, y['Class'].values)
dt.visualise('unpruned')
dt.cost_complexity_pruning(X, y['Class'].values)

# dt = cart.construct_tree(X_train, y_train)
# dt.populate_samples(X_train, y_train.values)
# dt.visualise('unpruned')
# print 'Unpruned accuracy:', accuracy_score(y_test, dt.evaluate_multiple(X_test), normalize=1)
#
# dt.cost_complexity_pruning(X_train, y_train.values)