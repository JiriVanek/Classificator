'''
Created on 11. 3. 2018

@author: Anet
'''
import random

def mix_data(X, y):
    
    if(len(X) != len(y)):
        print("Data not consistence: ")
    
    else:
        
        for i in range(1,len(X)):
            j = random.randrange(0, len(X),2);
            ax = X[i - 1];
            X[i - 1] = X[j];
            X[j] = ax;
            
            ay = y[i - 1];
            y[i - 1] = y[j];
            y[j] = ay;
            
            
            