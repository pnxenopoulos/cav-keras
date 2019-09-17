""" Concept Activation Vectors in Python """

import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.models import Sequential
from keras.layers import InputLayer, Reshape, Flatten
from sklearn.linear_model import SGDClassifier


class TCAV(object):
    """ Class for concept activation vectors for Keras models.

    Attributes:
        model: A Sequential Keras model
        model_f: A Sequential Keras model that is the first half of model
        model_h: A Sequential Keras model that is the second half of model
        cav: A numpy array containing the concept activation vector
        sensitivity: A numpy array containing sensitivities
        tcav_score: A list of the TCAV scores for the classes
        y_labels: A numpy array containing class labels
    """

    def __init__(self, model=None):
        """ Initialize the class with empty variables
        """
        self.model = model
        self.model_f = None
        self.model_h = None
        self.cav = None
        self.sensitivity = None
        self.tcav_score = []
        self.y_labels = None

    def set_model(self, model=None):
        """ Function to set the model for the TCAV object

        Args:
            model: A Keras 'Sequential' model

        Raises:
            ValueError: If the model is not of the Keras Sequential type
        """
        self.model = model

    def split_model(self, bottleneck, conv_layer=True):
        """ Split the model on a given bottleneck layer

        Args:
            bottleneck: An integer specifying which layer to split the model
            conv_layer: A Boolean value specifying if we are splitting on a convolutional layer

        Returns:
            model_f: The model containing up to layer 'bottleneck'
            model_h: The model containing from layer 'bottleneck' to the end

        Raises:
            ValueError: If the bottleneck layer value is less than 0 or greater than the total number of layers
            Warning: If the bottleneck layer is a convolutional layer
        """
        if bottleneck < 0 or bottleneck >= len(self.model.layers):
            raise ValueError(
                "Bottleneck layer must be greater than or equal to 0 and less than the number of layers!"
            )
        self.model_f = Sequential()
        self.model_h = Sequential()
        for current_layer in range(0, bottleneck + 1):
            self.model_f.add(self.model.layers[current_layer])
        if conv_layer:
            self.model_f.add(Flatten())
            self.model_h.add(
                InputLayer(
                    input_shape=self.model_f.layers[bottleneck + 1].output_shape[1:]
                )
            )
            self.model_h.add(Reshape(self.model.layers[bottleneck + 1].input_shape[1:]))
            for current_layer in range(bottleneck + 1, len(self.model.layers)):
                self.model_h.add(self.model.layers[current_layer])
        else:
            self.model_h.add(
                InputLayer(
                    input_shape=self.model.layers[bottleneck + 1].input_shape[1:]
                )
            )
            for current_layer in range(bottleneck + 1, len(self.model.layers)):
                self.model_h.add(self.model.layers[current_layer])

    def _create_counterexamples(self, x_concept):
        """ A function to create random counterexamples

        Args:
            x_concept: The training concept data

        Return:
            counterexamples: A numpy array of counterexamples
        """
        n = x_concept.shape[0]
        height = x_concept.shape[1]
        width = x_concept.shape[2]
        channels = x_concept.shape[3]
        counterexamples = []
        for i in range(0, n):
            counterexamples.append(
                np.rint(np.random.rand(height, width, channels) * 255)
            )
        return np.array(counterexamples)

    def train_cav(self, x_concept):
        """ Calculate the concept activation vector

        Args:
            x_concept: A numpy array of concept training data

        Returns:
            cav: A concept activation vector
        """
        counterexamples = self._create_counterexamples(x_concept)
        x_train_concept = np.append(x_concept, counterexamples, axis=0)
        y_train_concept = np.repeat([1, 0], [x_concept.shape[0]], axis=0)
        concept_activations = self.model_f.predict(x_train_concept)
        lm = SGDClassifier(
            loss="perceptron", eta0=1, learning_rate="constant", penalty=None
        )
        lm.fit(concept_activations, y_train_concept)
        coefs = lm.coef_
        self.cav = np.transpose(-1 * coefs)

    def calculate_sensitivity(self, x_train, y_train):
        """ Calculate and return the sensitivity

        Args:
            x_train: A numpy array of the training data
            y_train: A numpy array of the training labels
        """
        model_f_activations = self.model_f.predict(x_train)
        reshaped_labels = np.array(y_train).reshape((x_train.shape[0], 1))
        tf_y_labels = tf.convert_to_tensor(reshaped_labels, dtype=np.float32)
        loss = k.binary_crossentropy(tf_y_labels, self.model_h.output)
        grad = k.gradients(loss, self.model_h.input)
        gradient_func = k.function([self.model_h.input], grad)
        calc_grad = gradient_func([model_f_activations])[0]
        sensitivity = np.dot(calc_grad, self.cav)
        self.sensitivity = sensitivity
        self.y_labels = y_train

    def print_sensitivity(self):
        """ Print the sensitivities in a readable way
        """
        if type(self.y_labels) == list:
            self.y_labels = np.array(self.y_labels)
        print(
            "The sensitivity of class 1 is ",
            str(
                np.sum(self.sensitivity[np.where(self.y_labels == 1)[0]] > 0)
                / np.where(self.y_labels == 1)[0].shape[0]
            ),
        )
        print(
            "The sensitivity of class 0 is ",
            str(
                np.sum(self.sensitivity[np.where(self.y_labels == 0)[0]] > 0)
                / np.where(self.y_labels == 0)[0].shape[0]
            ),
        )
