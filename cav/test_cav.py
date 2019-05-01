import unittest

import numpy as np

from keras.initializers import Constant
from keras.models import Sequential
from keras.layers import Dense

from cav import *

class TestModelSplit(unittest.TestCase):
    ''' Tests for model split function
    '''
    def test_length(self):
        ''' Test that the length of the output is 2
        '''
        model = Sequential()
        model.add(Dense(50, input_dim = 100))
        model.add(Dense(40))
        model.add(Dense(30))
        model.add(Dense(20))
        model.add(Dense(10))
        model_f, model_h = return_split_models(model, 2)
        self.assertEqual(len(model_f.layers), 3)
        self.assertEqual(len(model_h.layers), 2)

    def test_model_h_input(self):
        ''' Test that model_h has correct input dimensions
        '''
        model = Sequential()
        model.add(Dense(50, input_dim = 100))
        model.add(Dense(40))
        model.add(Dense(30))
        model.add(Dense(20))
        model.add(Dense(10))
        model_f, model_h = return_split_models(model, 2)
        self.assertEqual(model_h.input_shape, (None, 30))

class TestBinaryClassifier(unittest.TestCase):
    ''' Tests for CAV function
    '''
    def test_dimensions(self):
        ''' Test the dimensions of the trained CAV
        '''
        model = Sequential()
        model.add(Dense(5, kernel_initializer=Constant(value=9), input_dim = 5))
        model.add(Dense(4, kernel_initializer=Constant(value=9)))
        model.add(Dense(3, kernel_initializer=Constant(value=9)))
        model.add(Dense(2, kernel_initializer=Constant(value=9)))
        model.add(Dense(1, kernel_initializer=Constant(value=9)))
        model_f, model_h = return_split_models(model, 2)
        x_concept = np.array([[6,7,8,9,10], [5,6,7,8,9], [1,2,3,4,5], [2,3,4,5,6]])
        y_concept = [1, 1, 0, 0]
        cav = train_cav(model_f, x_concept, y_concept)
        self.assertEqual(cav.shape[0], 3)

    def test_cav_output(self):
        ''' Test the output of the trained CAV
        '''
        np.random.seed(1996)
        model = Sequential()
        model.add(Dense(5, kernel_initializer=Constant(value=9), input_dim = 5))
        model.add(Dense(4, kernel_initializer=Constant(value=9)))
        model.add(Dense(3, kernel_initializer=Constant(value=9)))
        model.add(Dense(2, kernel_initializer=Constant(value=9)))
        model.add(Dense(1, kernel_initializer=Constant(value=9)))
        model_f, model_h = return_split_models(model, 2)
        x_concept = np.array([[6,7,8,9,10], [5,6,7,8,9], [1,2,3,4,5], [2,3,4,5,6]])
        y_concept = [1, 1, 0, 0]
        cav = train_cav(model_f, x_concept, y_concept)
        cav_rounded = np.round(cav)
        self.assertEqual(np.sum(np.array([[-1], [0], [0]]) == cav_rounded), 3)

if __name__ == '__main__':
    unittest.main()
