'''
Created on 9. 3. 2018

@author: Aneta Medunova

Trida nacte nazvy souboru pro testovani a jejich targetove stimuly

'''


def load_training_data_names():
    training_files = []
    s = open('subject_target_train',"r")
    for line in s:
        training_files.append((line[:-5],line[-4],line[-2]))

    return training_files

def load_testing_data_names():
    testing_files = []
    s = open('subject_target_test',"r")
    for line in s:
        testing_files.append((line[:41],line[42:]))

    return testing_files

