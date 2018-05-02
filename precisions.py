'''
Created on 4. 4. 2018

@author: Anet
'''

def accuracy (mapResults, true_label, number_of_all, instruction):
    if(instruction == 0):
        true = mapResults.count('R1') + mapResults.count('R2') + mapResults.count('R3') + mapResults.count('C1') + mapResults.count('C2') + mapResults.count('C3')
    else:
        true = mapResults.count('door') + mapResults.count('window') + mapResults.count('radio') + mapResults.count('lamp') + mapResults.count('phone') + mapResults.count('tv') + mapResults.count('food') + mapResults.count('toilet') + mapResults.count('helps')
    
    
    true_positive = mapResults.count(true_label)
    false_positive = true - true_positive
    true_negative = number_of_all - false_positive - true_positive
    accuracy = (true_positive + true_negative)/number_of_all
    
    accuracy_round = round((accuracy*100), 2)
    print("Accuracy: ",accuracy_round,"%")
    
    
def precision(mapResults,true_label, instruction):
    if(instruction == 0):
        true = mapResults.count('R1') + mapResults.count('R2') + mapResults.count('R3') + mapResults.count('C1') + mapResults.count('C2') + mapResults.count('C3')
    else:
        true = mapResults.count('door') + mapResults.count('window') + mapResults.count('radio') + mapResults.count('lamp') + mapResults.count('phone') + mapResults.count('tv') + mapResults.count('food') + mapResults.count('toilet') + mapResults.count('helps')
    
    
    true_positive = mapResults.count(true_label)
    false_positive = true - true_positive
   
    if((true_positive + false_positive) != 0):
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0
    precision_round = round((precision*100),2)
    print("Precision is: ",precision_round,"%")

def recall(mapResults,true_label, instruction):
    if(instruction == 0):
        expected_targets = 20
    else:
        expected_targets = 15 
        
    true_positive = mapResults.count(true_label)
    false_negative = expected_targets - true_positive
    if((true_positive + false_negative) != 0):
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0
        
    recall_round = round((recall*100),2)
    print("Recall is: ",recall_round,"%")
    