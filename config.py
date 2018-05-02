'''
Created on 5. 4. 2018

@author: Anet
'''

from keras.models import Sequential
'''
Config at main class

'''
# input data to training
instruction_files_count = 5
matrix_files_count = 10

# input data to predict
instruction_files_to_pred = 5
matrix_files_to_pred = 10




# filter frequency
low_filter_frequency = 0.1
high_filter_frequency = 30.0

# epochs constants
epoch_tmin = -0.1
epoch_tmax = 1.0

# baseline corection constant
baseline_min = -0.1
baseline_max = 0.0



# Create evoked structure
conditions = ["door", "window", "radio", "lamp", "phone", "tv", "food", "toilet", "helps"]

# Set EEG event list - instruction
event_id_instruction = {'door': 1,'window': 2,'radio': 3,'lamp': 4,'phone': 5,'tv': 6,'food': 7,'toilet': 8,'helps': 9}
event_id_matrix = {'R1': 1,'R2': 2,'R3': 3,'C1': 4,'C2': 5,'C3': 6}

# Set color of events
color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c', 5: 'black', 6: 'blue',7: 'magenta',8: 'pink',9: 'brown'}


# Plot chart 
colors = dict(door="Green", window="Yellow", radio="Red", lamp="Crimson", phone="Black", tv="Blue", food="Pink", toilet="CornFlowerBlue", helps="CornFlowerBlue",)
linestyles = dict(phone='--', toilet='--', lamp='--', radio='-', window='-', food='-', door='-', helps='-', tv='-')


chan = ("Fp1","Fp2","Fz","Cz","Pz")


'''
Config at feature class
'''


'''
Print results
'''

labels = ['door','window','radio','lamp','phone','tv','food','toilet','helps']
mark_length = 9

labels_matrix = ['R1','R2','R3','C1','C2','C3']
mark_length_matrix = 6



model = Sequential()













