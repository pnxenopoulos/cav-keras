import numpy as np
import keras
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def returnModelLayer(model, layer):
    '''
    '''
    model_cut = Sequential()
    for l in range(0,layer+1):
        model_cut.add(model.layers[l])
    return model_cut

def returnLayerActivations(model_cut, x):
    '''
    '''
    activations = model_cut.predict(x)
    # maybe flatten
    return activations

def trainLinearClassifier(activations, y_concept, opt):
    '''
    '''
    binary_linear_classifier = Sequential()
    binary_linear_classifier.add(Dense(2, input_shape=activations.shape[1:], activation='softmax'))
    binary_linear_classifier.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    binary_linear_classifier.fit(activations, y_concept, batch_size=32, epochs=10, shuffle=True, verbose=1)
    return binary_linear_classifier

def conceptSensitivity(x, binary_linear_classifier, activations, cav_vector):
    '''
    '''
    

def prepCIFAR10(batch_size = 32, epochs = 20, optimizer = keras.optimizers.Adam(lr=0.001)):
    '''
    '''
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    # Start
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    # train the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True, verbose=1)
    return x_train, y_train, model
