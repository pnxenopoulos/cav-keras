import unittest

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

if __name__ == '__main__':
    unittest.main()
