''' Utilities for concept activation vectors '''
import numpy as np
import tensorflow as tf

from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from sklearn.linear_model import SGDClassifier

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
    # Write input layer for model_h
    model_h.add(InputLayer(input_shape=model.layers[layer+1].input_shape[1:]))
    for current_layer in range(layer+1, len(model.layers)):
        model_h.add(model.layers[current_layer])
    return model_f, model_h

def train_cav(model_f, x_concept, y_concept):
    ''' Return the concept activation vector for the concept

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
    cav : (numpy.ndarray)
        Concept activation vector
    '''
    concept_activations = model_f.predict(x_concept)
    lm = SGDClassifier(loss = "perceptron", eta0=  1, learning_rate = "constant", penalty = None)
    lm.fit(concept_activations, y_concept)
    coefs = lm.coef_
    cav = np.transpose(-1 * coefs)
    return cav

def conceptual_sensitivity(examples, example_labels, model_f, model_h, concept_cav):
    ''' Return the conceptual conceptual sensitivity for a given example

    Parameters
    ----------
    examples : (numpy.ndarray)
        Examples to calculate the concept sensitivity (be sure to reshape)
    example_labels : (numpy.ndarray)
        Corresponding example labels
    model_f : (keras.engine.sequential.Sequential)
        First Keras sequential model from return_split_models()
    model_h : (keras.engine.sequential.Sequential)
        Second Keras sequential model from return_split_models()
    concept_cav : (numpy.ndarray)
        Numpy array with the linear concept activation vector for a given concept

    Returns
    -------
    sensitivity : (numpy.ndarray)
        Array of sensitivities for specified examples
    '''
    model_f_activations = model_f.predict(examples)
    reshaped_labels = np.array(example_labels).reshape((examples.shape[0], 1))
    tf_example_labels = tf.convert_to_tensor(reshaped_labels, dtype=np.float32)
    loss = k.mean(k.binary_crossentropy(tf_example_labels, model_h.output))
    grad = k.gradients(loss, model_h.input)
    gradient_func = k.function([model_h.input], grad)
    calc_grad = gradient_func([model_f_activations])[0]
    sensitivity = np.dot(calc_grad, concept_cav)
    return sensitivity

def tcav_score(x_train, y_train, model, layer, x_concept, y_concept):
    ''' Returns the TCAV score for the training data to a given concept

    Parameters
    ----------
    x_train : (numpy.ndarray)
        Training data where the i-th entry as x_train[i] is one example
    y_train : (numpy.ndarray)
        Training labels where the i-th entry as y_train[i] is one example
    model : (keras.engine.sequential.Sequential)
        Trained model to use
    layer : (int)
        Integer specifying layer to split model on
    x_concept : (numpy.ndarray)
        Training data for concept set, has same size as model training data
    y_concept : (numpy.ndarray)
        Labels for concept set, has same size as model training labels

    Returns
    -------
    tcav : (list)
        TCAV score for given concept and class
    '''
    model_f, model_h = return_split_models(model, layer)
    concept_cav = train_cav(model_f, x_concept, y_concept)
    unique_labels = np.unique(y_train)
    tcav = []
    for label in unique_labels:
        training_subset = x_train[np.array(y_train) == 1]
        set_size = training_subset.shape[0]
        count_of_sensitivity = 0
        for example in training_subset:
            sensitivity = conceptual_sensitivity(example, model_f, model_h, concept_cav)
            if sensitivity > 0:
                count_of_sensitivity = count_of_sensitivity + 1
        tcav.append(count_of_sensitivity/set_size)
    return tcav

def create_counterexamples(n = 500, height = 32, width = 32, channels = 3):
    ''' Returns a list of counterexamples given from the concept training set

    Parameters
    ----------
    n : (int)
        Number of counterexamples to generate
    height : (int)
        Height of image of counterexamples
    width : (int)
        Width of image of counterexamples
    channels : (int)
        Number of channels to generate

    Returns
    -------
    counterexamples : (list)
        Array of counterexamples
    '''
    counterexamples = []
    for i in range(0,n):
        image = np.rint(np.random.rand(height, width, channels) * 255)
        counterexamples.append(image)
    counterexamples = np.array(counterexamples)
    return counterexamples
