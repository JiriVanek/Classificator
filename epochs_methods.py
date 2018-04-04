'''
Created on 9. 3. 2018

@author: Anet
'''
import stringprep as s
from _cffi_backend import string

#metoda vrrátí pole epoch, které obsahují pouze targetové (nontargetové) podle znacky na vstupu
def filter_epochs_target(epochs, events, target):
    # Projde udalosti a vybira poze targetove udaje
    
    event_id_t = ['{\'door\': 1}','{\'window\': 2}','{\'radio\': 3}','{\'lamp\': 4}','{\'phone\': 5}','{\'tv\': 6}','{\'food\': 7}','{\'toilet\': 8}','{\'helps\': 9}']

    epochs_t = []
    
    target = int(target)
    target -= 1
    
    
    for j in range(len(epochs)):
    
        a = epochs[j].event_id
        a = str(a)
        b = event_id_t[target]
        
        if(a == b):
          
            epochs_t.append(epochs[j])
            
#     print(epochs_t)  
    return epochs_t
