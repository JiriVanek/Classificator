'''
Created on 9. 3. 2018

@author: Anet
'''
# -*- coding: utf-8 -*-
# from asn1crypto.core import range
"""
Created on Thu Mar  8 21:53:19 2018

@author: Anet
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def solve (X,y,X_pred):
#     plt.plot(np.mean(X[y==0], axis = 1))
#     plt.show()
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
            solver='svd', store_covariance=False, tol=0.0001)
    
    print("Predict: ")
    
#     X_pred = np.reshape(X_pred,(1,-1))
#     print(clf.predict(X_pred))
    

    x_event = []
    for i in range(len(X_pred)):
        
        
        X_pred[i] = np.reshape(X_pred[i],(1,-1))
#          print(clf.predict(X_pred[i]))
        x_event.append(clf.predict(X_pred[i]))
      
    x_event = np.array(x_event)
    return x_event
    

