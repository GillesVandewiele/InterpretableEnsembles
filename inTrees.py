# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys

from sklearn.externals import joblib

from constructors.c45orangeconstructor import C45Constructor
from data.load_datasets import load_heart

sys.path.append('../')
from defragTrees import *

import numpy as np
import re
from sklearn import tree
from sklearn.grid_search import GridSearchCV

import pylab as pl
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
import os

from rpy2.robjects.packages import importr
import pandas.rpy.common as com
import rpy2.robjects as ro

#************************
# inTree Class
#************************
class inTreeModel(RuleModel):
    def __init__(self, modeltype='regression'):
        super(inTreeModel, self).__init__(modeltype=modeltype)
    
    #************************
    # Fit and Related Methods
    #************************
    def fit(self, y, X, filename, featurename=[]):
        self.dim_ = X.shape[1]
        self.setfeaturename(featurename)
        self.setdefaultpred(y)
        if self.modeltype_ == 'regression':
            v1 = np.percentile(y, 17)
            v2 = np.percentile(y, 50)
            v3 = np.percentile(y, 83)
            val = (v1, v2, v3)
        mdl = self.__parsInTreesFile(filename)
        for m in mdl:
            if m[3] == 'X[,1]==X[,1]':
                self.rule_.append([])
            else:
                subrule = []
                ll = m[3].split(' & ')
                for r in ll:
                    id1 = r.find(',') + 1
                    id2 = r.find(']')
                    idx = int(r[id1:id2])
                    if '>' in r:
                        v = 1
                        id1 = r.find('>') + 1
                        t = float(r[id1:])
                    else:
                        v = 0
                        id1 = r.find('<=') + 2
                        t = float(r[id1:])
                    subrule.append((idx, v, t))
                self.rule_.append(subrule)
            if self.modeltype_ == 'classification':
                self.pred_.append(int(m[4]))
            elif self.modeltype_ == 'regression':
                if m[4] == 'L1':
                    self.pred_.append(val[0])
                elif m[4] == 'L2':
                    self.pred_.append(val[1])
                elif m[4] == 'L3':
                    self.pred_.append(val[2])
        
    def __parsInTreesFile(self, filename):
        f = open(filename)
        line = f.readline()
        mdl = []
        while line:
            if not'[' in line:
                line = f.readline()
                continue
            id1 = line.find('[') + 1
            id2 = line.find(',')
            idx = int(line[id1:id2])
            if idx > len(mdl):
                mdl.append(re.findall(r'"([^"]*)"', line))
            else:
                mdl[idx-1] += re.findall(r'"([^"]*)"', line)
            line = f.readline()
        f.close()
        return mdl

#************************
# DTree Class
#************************
class DTreeModel(RuleModel):
    def __init__(self, modeltype='regression', max_depth=[None, 2, 4, 6, 8], min_samples_leaf=[5, 10, 20, 30], cv=5):
        super(DTreeModel, self).__init__(modeltype=modeltype)
        self.max_depth_ = max_depth
        self.min_samples_leaf_ = min_samples_leaf
        self.cv_ = cv
        
    #************************
    # Fit and Related Methods
    #************************
    def fit(self, y, X, featurename=[]):
        self.dim_ = X.shape[1]
        self.setfeaturename(featurename)
        self.setdefaultpred(y)
        param_grid = {"max_depth": self.max_depth_, "min_samples_leaf": self.min_samples_leaf_}
        if self.modeltype_ == 'regression':
            mdl = tree.DecisionTreeRegressor()
        elif self.modeltype_ == 'classification':
            mdl = tree.DecisionTreeClassifier()
        grid_search = GridSearchCV(mdl, param_grid=param_grid, cv=self.cv_)
        grid_search.fit(X, y)
        mdl = grid_search.best_estimator_
        self.__parseTree(mdl)
        
    def __parseTree(self, mdl):
        t = mdl.tree_
        m = len(t.value)
        left = t.children_left
        right = t.children_right
        feature = t.feature
        threshold = t.threshold
        value = t.value
        parent = [-1] * m
        ctype = [-1] * m
        for i in range(m):
            if not left[i] == -1:
                parent[left[i]] = i
                ctype[left[i]] = 0
            if not right[i] == -1:
                parent[right[i]] = i
                ctype[right[i]] = 1
        for i in range(m):
            if not left[i] == -1:
                continue
            subrule = []
            c = ctype[i]
            idx = parent[i]
            while not idx == -1:
                subrule.append((feature[idx], c, threshold[idx]))
                c = ctype[idx]
                idx = parent[idx]
            self.rule_.append(subrule)
            if np.array(value[i]).size > 1:
                self.pred_.append(np.argmax(np.array(value[i])))
            else:
                self.pred_.append(np.asscalar(value[i]))


#************************
# Plot Methods
#************************
def plotRule(mdl, X, d1, d2, alpha=0.8, filename='', rnum=-1):
    cmap = cm.get_cmap('cool')
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[19, 1]})
    if rnum <= 0:
        rnum = len(mdl.rule_)
    else:
        rnum = min(len(mdl.rule_), rnum)
    for i in range(rnum):
        r = mdl.rule_[i]
        box, vmin, vmax = __r2boxWithX(r, X)
        if mdl.modeltype_ == 'regression':
            c = cmap(mdl.pred_[i])
        elif mdl.modeltype_ == 'classification':
            r = mdl.pred_[i] / (np.unique(mdl.pred_).size - 1)
            c = cmap(r)
        ax1.add_patch(pl.Rectangle(xy=[box[0, d1], box[0, d2]], width=(box[1, d1] - box[0, d1]), height=(box[1, d2] - box[0, d2]), facecolor=c, linewidth='2.0', alpha=alpha))
    ax1.set_xlabel('x1', size=22)
    ax1.set_ylabel('x2', size=22)
    ax1.set_title('Simplified Model (K = %d)' % (rnum,), size=28)
    colorbar.ColorbarBase(ax2, cmap=cmap, format='%.1f')
    ax2.set_ylabel('Predictor z', size=22)
    plt.show()
    if not filename == '':
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close()

def __r2boxWithX(r, X):
    vmin = np.min(X, axis=0)
    vmax = np.max(X, axis=0)
    box = np.c_[vmin, vmax].T
    for rr in r:
        if rr[1] == 0:
            box[1, rr[0]-1] = np.minimum(box[1, rr[0]-1], rr[2])
        else:
            box[0, rr[0]-1] = np.maximum(box[0, rr[0]-1], rr[2])
    return box, vmin, vmax

'''
  left daughter right daughter split var split point status prediction
1             2              3        14         4.5      1          0
2             0              0         0         0.0     -1          1
3             0              0         0         0.0     -1          2

ruleExec <- unique(extractRules(treeList, X2))
ruleMetric <- getRuleMetric(ruleExec, X2, y2)
ruleMetric <- pruneRule(ruleMetric, X2, y2)
learner <- buildLearner(ruleMetric, X2, y2)
out <- capture.output(learner)
'''


def tree_to_R_object(tree, features):
    node_mapping = {}
    feature_mapping = {}
    nodes = tree.get_nodes()
    nodes.extend(tree.get_leaves())
    for i, node in enumerate(nodes):
        node_mapping[node] = i+1
    for i, feature in enumerate(features):
        feature_mapping[feature] = i+1
    vectors = []
    for node in nodes:
        if node.value is not None:
            vectors.append([node_mapping[node], node_mapping[node.left], node_mapping[node.right],
                            feature_mapping[node.label], node.value, 1, 0])
        else:
            vectors.append([node_mapping[node], 0, 0, 0, 0.0, 1, node.label])

    df = pd.DataFrame(vectors)
    df.columns = ['id', 'left daughter', 'right daughter', 'split var', 'split point', 'status', 'prediction']
    df = df.set_index('id')
    df.index.name = None
    return com.convert_to_r_dataframe(df), feature_mapping


heart, features, label_col, dataset_name = load_heart()
heart = heart.iloc[np.random.permutation(len(heart))].reset_index(drop=True)
heart_train = heart.head(int(0.75*len(heart)))
heart_test = heart.tail(int(0.25*len(heart)))

X_train = heart_train.drop(label_col, axis=1)
y_train = heart_train[label_col]
X_test = heart_test.drop(label_col, axis=1)
y_test = heart_test[label_col]

c45 = C45Constructor()
c45_tree = c45.construct_tree(X_train, y_train)
r_object, features_map = tree_to_R_object(c45_tree, features)

rfImport = importr('randomForest')
inTreesImport = importr('inTrees')

ro.r('data(iris);')
ro.globalenv["X"] = com.convert_to_r_dataframe(X_train)
ro.globalenv["target"] = ro.FactorVector(y_train.values.tolist())
# ro.r('print(target)')
# ro.r('print(iris[,"Species"]);')
ro.r('rf <- randomForest(X, target)')

ro.r('treeList <- {}')
ro.globalenv["treeList$nTree"] = 1
ro.globalenv["treeList$list"] = [r_object]
ro.r('print(treeList)')
ro.r('print(RF2List(rf)[1])')
ro.r('print(RF2List(rf)$list[[351]])')

# ro.r('print(extractRules(treeList, X))')

# ro.globalenv["treeList"] = [r_object]
# ro.globalenv["X"] = com.convert_to_r_dataframe(X_train)
# print list(y_train.values)
# ro.globalenv["target"] = list(y_train.values)
# ro.r('rf <- randomForest(X,as.factor(target))')
#
#
# ro.r('print(extractRules(treeList, X))')


# ro.r(r_object)
# ro.r('print(r_object)')
# ro.r('ruleExec <- unique(extractRules(treeList, X2))')

# heart, features, class_col, dataset_name = load_heart()
# heart = heart.iloc[np.random.permutation(len(heart))].reset_index(drop=True)
# heart_train = heart.head(int(0.75*len(heart)))
# heart_test = heart.tail(int(0.25*len(heart)))
# heart_train.to_csv('train.csv', header=False)
# heart_test.to_csv('test.csv', header=False)
# trfile = 'train.csv'
# tefile = 'test.csv'
# Ztr = np.loadtxt(trfile, delimiter=',')
# Xtr = Ztr[:, :-1]
# ytr = Ztr[:, -1]
# Zte = np.loadtxt(tefile, delimiter=',')
# Xte = Zte[:, :-1]
# yte = Zte[:, -1]
# #
# #
# os.system('Rscript buildClfForest.R %s %s ./result_%s %d 0' % (trfile, tefile, 'test', 5))
# #
# prefix = 'test'
# modeltype = 'classification'
# print('test')
# mdl2 = inTreeModel(modeltype=modeltype)
# mdl2.fit(ytr, Xtr, './result_%s/inTrees.txt' % (prefix,), featurename=features)
# # joblib.dump(mdl2, '%s/%s_inTrees.mdl' % (dirname, prefix), compress=9)
# score, cover = mdl2.evaluate(yte, Xte, rnum=-1)
# print()
# print('<< inTrees >>')
# print('----- Evaluated Results -----')
# print('Test Error = %f' % (score,))
# print('Test Coverage = %f' % (cover,))
# print()
# # print(mdl2)
# plotRule(mdl2, Xtr, 0, 1, filename='%s_inTrees.pdf' % (prefix), rnum=-1)