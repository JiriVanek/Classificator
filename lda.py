"""
Created on Thu Mar  8 21:53:19 2018

@author: Anet
"""
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def solve (x_train,y_train,x_test):

    x_train = np.reshape(x_train,(-1, 100))

    clf = LinearDiscriminantAnalysis()
    clf.fit(x_train, y_train)
    LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
            solver='svd', store_covariance=False, tol=0.0001)
    
    
    x_event = []
    for i in range(len(x_test)):
        
        x_test[i] = np.reshape(x_test[i],(1,-1))
        x_event.append(clf.predict(x_test[i]))
      
    x_event = np.array(x_event)
    return x_event
    

