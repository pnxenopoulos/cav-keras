''' Utilities for concept activation vectors '''

from keras.models import Sequential
from keras.layers import Dense

def return_split_models(model, layer):
    ''' Split a model into model_f and model_h

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

def train_concept_classifier(model_f, x_concept, y_concept):
    ''' Train the binary classifier for the concept

    Parameters
    ----------
    model_f : (keras.engine.sequential.Sequential)
        First Keras sequential model from return_split_models()
    x_concept : (numpy.ndarray)
        Training data for concept set, has same size as model training data
    y_concept : (numpy.ndarray)
        Labels for concept set, has same size as model training labels

    Returns
    -------
    binary_classifier : (keras.engine.sequential.Sequential)
        Binary classifier Keras model
    '''
    concept_activations = model_f.predict(x_concept)
    binary_classifier = Sequential()
    binary_classifier.add(Dense(2, input_shape=concept_activations.shape[1:], activation='sigmoid'))
    binary_classifier.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    binary_classifier.fit(concept_activations, y_concept, batch_size=32, epochs=20, shuffle=True)
    return binary_classifier
