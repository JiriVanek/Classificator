'''
Created on 23. 3. 2018

@author: Anet
'''

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import numpy as np
from keras.callbacks import EarlyStopping
import plot_training



def solve(x_train,y_train, x_predict):


#     y_binary = to_categorical(y_train)
 
    x_predict = np.reshape(x_predict,(-1, 100))
   
    model = Sequential()

    input_dim = x_train.shape[1]
    

    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, input_dim=input_dim, activation='sigmoid'))
    
    earlyStopping = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')
    
    # For a binary classification problem
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])



    
    model_history = model.fit(x_train, y_train, epochs=100, batch_size=32)
    
    

    classes = model.predict(x_predict, batch_size=32)
    
    for i in range(len(classes)):
        for j in range(len(classes[i])):
            classes[i][j] = round(classes[i][j])
   
#     print(classes[:135,0])
#     plot_training.display_history(model_history)
    return classes
    

    