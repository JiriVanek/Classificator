import precisions as precision
import config


def print_guess(guess_results, epochs_to_predict, true_prediction, instruction):
        
    
    mapResults = []
    for i in range(len(guess_results)):
        if(guess_results[i] == 1):
            result = ''
            result = ''.join(epochs_to_predict[i].event_id)
            mapResults.append(result)
    
    if(instruction == 1):
        for j in range(config.mark_length):
#             print(config.labels[j]," :",mapResults.count(config.labels[j]))
            print ('%-8s : %3d' % (config.labels[j], mapResults.count(config.labels[j])))
           
    else:
        for j in range(config.mark_length_matrix):
            print(config.labels_matrix[j]," :",mapResults.count(config.labels_matrix[j]))

    print()

  
    precision.accuracy(mapResults,true_prediction, len(epochs_to_predict),instruction)
    precision.precision(mapResults, true_prediction,instruction)
    precision.recall(mapResults, true_prediction,instruction)


    
    