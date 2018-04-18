'''
Created on 9. 3. 2018

@author: Aneta Medunova

Trida nacte nazvy souboru pro testovani a jejich targetove stimuly

'''
import config


def load_training_data_names():
    training_files = []
    s = open('subject_target_train',"r")
    count = 0;
    for line in s:
        if(count < config.instruction_files_count):
            training_files.append((line[:-5],line[-4],line[-2]))
            count += 1
        else:
            training_files.append((line[:-6],line[-4],line[-2]))
#             print(training_files)
                
        

    return training_files

def load_testing_data_names():
    testing_files = []
    s = open('subject_target_test',"r")
    for line in s:
        testing_files.append((line[:41],line[42:]))

    return testing_files

