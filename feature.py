'''
Created on 9. 3. 2018

@author: Anet
'''
import numpy as np
import math
def feature_vector(epochs):
    
#will average for 50 samples
    samples_to_average = 50
    electrods_count = 5
    
#    Baseline correction average of first 100   
#    print(np.mean((epochs.get_data()[0][0][:100])))
    
    features = []    

    min = 100
    max = 1100
    
    '''puvodne bylo 100 a 1100'''

#5x elektrody

    for i in range(electrods_count):
        for j in range(min, max, samples_to_average):
            from_index = j
            to_index = j + samples_to_average
            features.append(np.mean((epochs.get_data()[0][i][from_index:to_index])))


     
    counterPower = 0
    for k in range(len(features)):
        number = features[k]
        counterPower = counterPower + (number**2)
      
    counterPower = math.sqrt(counterPower)
    for l in range(len(features)):
              
        features[l]/counterPower  


    return features





