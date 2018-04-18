import precisions as precision
import config


def print_guess (guess_results, epochs_to_predict, true_prediction):
        
    mapResults = []
    for i in range(len(guess_results)):
        if(guess_results[i] == 1):
            result = ''
            result = ''.join(epochs_to_predict[i].event_id)
            mapResults.append(result)
    
    for j in range(config.mark_length):
        print(config.labels[j]," :",mapResults.count(config.labels[j]))


    print()

  
    precision.accuracy(mapResults,true_prediction, len(epochs_to_predict))
    precision.precision(mapResults, true_prediction)
    precision.recall(mapResults, true_prediction)


    
    