''' Utilities for concept activation vectors '''

from keras.models import Sequential

def return_split_models(model, layer):
    ''' Function to split a model into model_f and model_h

    Parameters
    ----------
    model : (keras.engine.sequential.Sequential)
        Keras sequential model to split
    layer : (int)
        Integer specifying layer to split model on

    Returns
    -------
    model_f : (keras.engine.sequential.Sequential)
        Keras sequential model that is the first part
    model_h : (keras.engine.sequential.Sequential)
        Keras sequential model that is the second part
    '''
    model_f, model_h = Sequential(), Sequential()
    for current_layer in range(0, layer+1):
        model_f.add(model.layers[current_layer])
    for current_layer in range(layer+1, len(model.layers)):
        model_h.add(model.layers[current_layer])
    return [model_f, model_h]
