import precisions as precision


def print_guess (guess_results, epochs_to_predict, true_prediction):
    
    mapResults = []
    for i in range(len(guess_results)):
        if(guess_results[i] == 1):
            result = ''
            result = ''.join(epochs_to_predict[i].event_id)
            mapResults.append(result)
    
    print('door:    ',mapResults.count('door'))
    print('window:  ',mapResults.count('window'))
    print('radio:   ',mapResults.count('radio'))
    print('lamp:    ',mapResults.count('lamp'))
    print('phone:   ',mapResults.count('phone'))
    print('tv:      ',mapResults.count('tv'))
    print('food:    ',mapResults.count('food'))
    print('toilet:  ',mapResults.count('toilet'))
    print('helps:   ',mapResults.count('helps'))
    print()

  
    precision.accuracy(mapResults,true_prediction, len(epochs_to_predict))
    precision.precision(mapResults, true_prediction)
    precision.recall(mapResults, true_prediction)


    
    