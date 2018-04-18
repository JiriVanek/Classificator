'''
Created on 23. 3. 2018

@author: Anet
'''

from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from keras.callbacks import EarlyStopping
import plot_training
import matplotlib.pyplot as plt
import config


def train(x,y):

    x_train = []
    x_valid = []
    y_train = []
    y_valid = []
    
#     separate validation set and training set
    for i in range(len(x)):
        
        if(i%5 == 0):
            x_valid.append(x[i])
            if(y[i] == 0):
                y_valid.append(0)
            else:
                y_valid.append(1)
            
        else:
            x_train.append(x[i])
            if(y[i] == 0):
                y_train.append(0)
            else:
                y_train.append(1)
                
                
                

   
    x_train = np.reshape(x_train,(-1,100))
    x_valid = np.reshape(x_valid,(-1,100))
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    
    plt.plot(np.mean(x_train[y_train==1], axis=0))
    plt.plot(np.mean(x_train[y_train==0], axis=0))
    plt.title('Training epochs')
    plt.legend(['target', 'non-target'], loc='lower left')
    
    plt.show()
    
    plt.plot(np.mean(x_valid[y_valid==1], axis=0))
    plt.plot(np.mean(x_train[y_train==0], axis=0))
    plt.title('Validation epochs')
    plt.legend(['target', 'non-target'], loc='lower left')
    
    plt.show()
    
    
    input_dim = x_train.shape[1]
    
    
#     model = Sequential()
    config.model.add(Dense(64, input_dim=input_dim, activation='relu'))
    config.model.add(Dropout(0.2))
    config.model.add(Dense(32, activation='relu'))
    config.model.add(Dropout(0.2))

    config.model.add(Dense(1, input_dim=input_dim, activation='sigmoid'))
    
    
    
    config.model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    config.model.summary()

    earlyStopping = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')
   
    model_history = config.model.fit(x_train, y_train, epochs=300, batch_size=4,shuffle = True, callbacks=[earlyStopping], validation_data=(x_valid,y_valid))
    loss_and_metrics = config.model.evaluate(x_valid, y_valid, batch_size=4)
   
    plot_training.display_history(model_history)
    print(loss_and_metrics)
  
  
    
    
def solve(x_predict):
    
    x_predict = np.reshape(x_predict,(-1, 100))
     
    classes = config.model.predict(x_predict, batch_size=8)
    
    for i in range(len(classes)):
        for j in range(len(classes[i])):
            classes[i][j] = round(classes[i][j])
   
    
    return classes
    

    