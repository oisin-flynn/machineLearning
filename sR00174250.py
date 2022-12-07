#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on a sunny day

@author: Oisín Flynn
@id: R00174250
@Cohort: SD3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_csv("humanDetails.csv", encoding="ISO-8859-1")
pd.set_option('display.max_rows', df.shape[0] + 1)


def Task1():
    flt = df[[' workclass', 'age', 'Income', 'native-country']].copy()

    # remove unknown cells
    flt[' workclass'] = flt[' workclass'].replace(' ?', np.nan)
    flt = flt.dropna(subset=[' workclass'])

    # remove values with 's'
    flt['age'] = flt['age'].str.replace(r's', '')
    flt['age'] = flt['age'].apply(pd.to_numeric, errors='coerce')

    # remove unknown countries
    flt['native-country'] = flt['native-country'].replace(' ?', np.nan)
    flt = flt.dropna(subset=['native-country'])

    # convert income to numeric
    flt.loc[flt['Income'].str.contains('<=50K', na=False), 'Income'] = 1
    flt.loc[flt['Income'].str.contains('>50K', na=False), 'Income'] = 2
    flt['Income'] = flt['Income'].apply(pd.to_numeric, errors='coerce')

    allCountries = np.unique(flt['native-country'].astype(str))
    dict2 = {}
    c = 1
    for ac in allCountries:
        dict2[ac] = c
        c = c + 1
    flt['native-country'] = flt['native-country'].map(dict2)

    allClasses = np.unique(flt[' workclass'].astype(str))
    dict3 = {}
    c2 = 1
    for ac in allClasses:
        dict3[ac] = c2
        c2 = c2 + 1
    flt[' workclass'] = flt[' workclass'].map(dict3)
    flt = flt.dropna()

    X = flt[[' workclass', 'age', 'native-country']]  # features
    y = flt[['Income']]  # target variables

    # print(flt[[' workclass']])

    # # cross validation
    kfold = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
    trI = []
    tsI = []
    # decision tree classifier
    tree_clf = tree.DecisionTreeClassifier()
    tree_clf.fit(X, y)

    default = tree_clf.get_depth()
    tree_tr = []
    tree_ts = []

    for i in range(2, default - 1):
        tree_clf = tree.DecisionTreeClassifier(max_depth=i + 1)
        for train, test in kfold.split(X, y):
            tree_clf.fit(X.iloc[train], y.iloc[train])
            trI.append(tree_clf.score(X.iloc[train], y.iloc[train]))
            tsI.append(tree_clf.score(X.iloc[test], y.iloc[test]))
        print(i + 1)
        tree_tr.append(np.mean(trI))
        tree_ts.append(np.mean(tsI))
        print("----------")
    plt.plot(tree_ts)
    plt.plot(tree_tr)
    print(tree_clf.score(X, y))
    plt.show()


Task1()
# def Task2():
#     flt = df[['hours-per-week', 'occupation ', 'age', 'relationship',
#               'Income']].copy()
#
#     # replace unknown with most frequent
#     flt['occupation '] = flt['occupation '].replace(' ?', np.nan)
#     flt['occupation '] = flt['occupation '].fillna(flt['occupation '].mode()[0])
#
#     # remove values with 's'
#     flt['age'] = flt['age'].str.replace(r's', '')
#     flt['age'] = flt['age'].apply(pd.to_numeric, errors='coerce')
#
#     # remove 'Other-relative'
#     flt['relationship'] = flt['relationship'].replace(' Other-relative', np.nan)
#     flt = flt.dropna(subset=['relationship'])
#
#     # remove 'hours-per-week' values only mentioned once
#     flt['hours-per-week'] = flt['hours-per-week'].replace(82, np.nan)
#     flt['hours-per-week'] = flt['hours-per-week'].replace(94, np.nan)
#     flt['hours-per-week'] = flt['hours-per-week'].replace(92, np.nan)
#     flt['hours-per-week'] = flt['hours-per-week'].replace(87, np.nan)
#     flt['hours-per-week'] = flt['hours-per-week'].replace(74, np.nan)
#     flt = flt.dropna(subset=['hours-per-week'])
#
#     # convert income to numeric with 2 values
#     flt.loc[flt['Income'].str.contains('<=50K', na=False), 'Income'] = 1
#     flt.loc[flt['Income'].str.contains('>50K', na=False), 'Income'] = 2
#     flt['Income'] = flt['Income'].apply(pd.to_numeric, errors='coerce')
#
#     # cross validation
#     kfold = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
#
#
#
# Task2()
# def Task3():
#     flt = df[['fnlwgt', 'education-num', 'age', 'hours-per-week']].copy()
#
#     # remove values with 's'
#     flt['age'] = flt['age'].str.replace(r's', '')
#     flt['age'] = flt['age'].apply(pd.to_numeric, errors='coerce')

# Task3()
