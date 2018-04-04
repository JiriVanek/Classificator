# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 06:23:26 2018

@author: Anet
"""

print(__doc__)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def plot_lda(X,y):
#    iris = datasets.load_iris()

#X = iris.data
#y = iris.target
#    target_names = iris.target_names
#     X.reshape(-1,1)
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2


    for color, i in zip(colors, [0, 1, 2]):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('LDA of IRIS dataset')

    plt.show()