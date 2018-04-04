'''
Created on 4. 4. 2018

@author: Anet
'''

def accuracy (mapResults, true_label, number_of_all):
    true = mapResults.count('door') + mapResults.count('window') + mapResults.count('radio') + mapResults.count('lamp') + mapResults.count('phone') + mapResults.count('tv') + mapResults.count('food') + mapResults.count('toilet') + mapResults.count('helps')
    true_positive = mapResults.count(true_label)
    false_positive = true - true_positive
    true_negative = number_of_all - false_positive - true_positive
    accuracy = (true_positive + true_negative)/number_of_all
    
    accuracy_round = round((accuracy*100), 2)
    print("Accuracy: ",accuracy_round,"%")
    
    
def precision(mapResults,true_label):
    true = mapResults.count('door') + mapResults.count('window') + mapResults.count('radio') + mapResults.count('lamp') + mapResults.count('phone') + mapResults.count('tv') + mapResults.count('food') + mapResults.count('toilet') + mapResults.count('helps')
    true_positive = mapResults.count(true_label)
    false_positive = true - true_positive
   
    precision = true_positive / (true_positive + false_positive)
    precision_round = round((precision*100),2)
    print("Precision is: ",precision_round,"%")

def recall(mapResults,true_label):
    true_positive = mapResults.count(true_label)
    false_negative = 15 - true_positive
    recall = true_positive / (true_positive + false_negative)
    recall_round = round((recall*100),2)
    print("Recall is: ",recall_round,"%")
    