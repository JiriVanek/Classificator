# -*- coding: utf-8 -*-

"""
Created on Wed Feb 14 22:22:09 2018

@author: Anet
print(__doc__)
"""
import mne
import numpy as np
from mne import io
from mne.datasets import sample

import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
import feature as ft
import lda as lda
import epochs_methods as epoch_met
import input_test_data as load_file_names
from builtins import print
import mix_data_x_y as mix
import matplotlib.pyplot as plt
import neural_network as neural_network
import print_results
from sqlalchemy.sql.expression import true

#turn off log
mne.set_log_level('ERROR')

"""

    Load data

"""
print("Learn system data: ")
print("Set number of input data: ")
# 
#number_idata = input('Number: ')

"""
read data to presict
""" 
sample_data_path = sample.data_path()
 
# Load name of file .vhdr - to predict
# input_file_name = input('Name of .vhdr file: ')

# Set folder - raw brainvision files
data_path = sample.data_path() + '/raw_data/'

# Set path to .vhdr data
# path = data_path + 'Blink_visual_instruction_22_12_17_04.vhdr'


raw_to_predict = load_file_names.load_testing_data_names()

tmin=-0.1
tmax=1

# Set EEG event list - instruction
event_id = {'door': 1,'window': 2,'radio': 3,'lamp': 4,'phone': 5,'tv': 6,'food': 7,'toilet': 8,'helps': 9}




# mapu, ve ktere jsou ulozeny nazvy trenovacich souboru a jejich targetove/non-targetove znacky
files_training_map = load_file_names.load_training_data_names()
# mapu, ve ktere jsou ulozeny nazvy testovacich souboru a jejich targetove nazvy
files_testing_map = load_file_names.load_testing_data_names()


# Nacte data na trenovni ze souboru v mape
raw = []
data_count = len(files_training_map)
for i in range(len(files_training_map)):
    path = data_path + (files_training_map[i][0])
    raw.append(io.read_raw_brainvision(vhdr_fname=path, preload=True))

# nacte data na testovani
raw_to_predict = []
true_prediction = []
data_count = len(files_testing_map)
for i in range(len(files_testing_map)):
    path = data_path + (files_testing_map[i][0])
    true_prediction.append(files_testing_map[i][1])
    true_prediction[i] = true_prediction[i].strip()
    raw_to_predict.append(io.read_raw_brainvision(vhdr_fname=path, preload=True))
    raw_to_predict[i].filter(0.1,30)

    
# zatim pouzije jen prvni soubor - zmenit pro vsechny
event_to_predict = raw_to_predict[0]._events
epochs_to_predict = mne.Epochs(raw_to_predict[0],event_to_predict, event_id=event_id, tmin=tmin, tmax=tmax,baseline=(tmin, 0.0), preload=True)

    
# Plot raw data
# raw[0].plot(block=True, lowpass=40, n_channels=5)

for i in range(data_count): 
    raw[i].info['bads'] = ['STI 014'] 
#    raw[i].plot(block=True, lowpass=40, n_channels=6)
#    print(raw[i].info['ch_names'])
"""

    Preprocessing

"""

# Filter data
for i in range(data_count):
    raw[i].filter(0.1,30)


# Find events
print("Finding events")
events = []
for i in range(data_count):
    events.append(raw[i]._events)

# Set color of events
color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c', 5: 'black', 6: 'blue',7: 'magenta',8: 'pink',9: 'brown'}


"""
for i in range(data_count):
    mne.viz.plot_events(events[i],raw[i].info['sfreq'], raw[i].first_samp, color=color)
    """

#extract epochs

#raw.set_eeg_reference('average', projection=True) 

epochs = []
epochs_targets = []
epochs_non_targets = []


# Vytvori epochy, z vytvorenych Epoch potom vybere ty targetove a ulozi je do epochs_target
for i in range(data_count):
    epochs.append(mne.Epochs(raw[i],events[i], event_id=event_id, tmin=tmin, tmax=tmax,baseline=(tmin, 0.0), preload=True))
    epochs_targets.append(epoch_met.filter_epochs_target(epochs[i], events[i], files_training_map[i][1]))  
    epochs_non_targets.append(epoch_met.filter_epochs_target(epochs[i], events[i], files_training_map[i][2]))  
            

"""
#print(len(epochs.events)) 
for i in range(data_count):
    mne.viz.plot_epochs(epochs[i])
"""
# print(epochs_non_targets[1][1])
# epochs.plot(title="Events epochs", n_epochs=(len(epochs.events)),event_colors=color)
# mne.viz.plot_epochs(epochs, title="Events epochs", n_epochs=15,event_colors=color)



# Create evoked structure
conditions = ["door", "window", "radio", "lamp", "phone", "tv", "food", "toilet", "helps"]

evoked_dict = [[]]

#    evoked_dict[i] = dict()
for i in range(data_count):
    evoked_dict.append('')
    evoked_dict[i] = dict()
    for condition in conditions:
        evoked_dict[i][condition] = epochs[i][condition].average()
       
#print(evoked_dict)

# Plot chart 
colors = dict(door="Green", window="Yellow", radio="Red", lamp="Crimson", phone="Black", tv="Blue", food="Pink", toilet="CornFlowerBlue", helps="CornFlowerBlue",)
linestyles = dict(phone='--', toilet='--', lamp='--', radio='-', window='-', food='-', door='-', helps='-', tv='-')

"""
for i in range(data_count):
    mne.viz.plot_compare_evokeds(evoked_dict[i], title="ERP chart", colors=colors, linestyles=linestyles, gfp=False)
"""
 
#print(evoked_dict)




"""

Extrakce priznaku

"""
# print(epochs_targets[0][0].ch_names)
labels = epochs[0].events[:, -1]

# print(epochs[0].get_data)
# print("")

# for ep in epochs[:2]:
#     print(ep)
    
#print(epochs[1].get_data())    


#feature extraction
    
target_features = []
non_target_features = []
X = []

#print(event_id.get("toilet"))

test_sample_count = 5
chan = ("Fp1","Fp2","Fz","Cz","Pz")


y = []
s = open('target_nontarget',"r")
# for line in s:
#     y.append(line[:1])


# Prepare data to training LDA
target_nontarget_epochs = epochs_targets + epochs_non_targets
# print(len(target_nontarget_epochs))

for i in range(len(target_nontarget_epochs)):
    #count of target epochs     
    for j in range(len(target_nontarget_epochs[i])):
    
        pick_epochs = target_nontarget_epochs[i][j].pick_channels(chan)
        X.append(ft.feature_vector(pick_epochs))
        if(i < epochs_targets.__len__()):
            y.append(1)
        else:
            y.append(0)


# Prepare data to predict 
X_pred = []
y = np.array(y)
for i in range(len(epochs_to_predict)):
    pick_epoch_to_predict = epochs_to_predict[i].pick_channels(chan)
    X_pred.append(ft.feature_vector(pick_epoch_to_predict))

mix.mix_data(X, y)


X = np.reshape(X,(-1, 100))
plt.plot(np.mean(X[y==1], axis=0))
plt.plot(np.mean(X[y==0], axis=0))
# plt.show()

data_path = sample.data_path()
for i in range(len(X_pred)):
    name = str(i)+'.png'
    plt.plot(X_pred[i])
#     plt.savefig(name)




# true_prediction = 'toilet'
x_event_neural = neural_network.solve(X,y,X_pred)
print("_______________ Neural network ______________")
print_results.print_guess(x_event_neural, epochs_to_predict, true_prediction[0])

x_event_lda = lda.solve(X,y,X_pred)
print("_______________ LDA ______________")
print_results.print_guess(x_event_lda, epochs_to_predict, true_prediction[0])




