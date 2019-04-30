import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from keras.models import Sequential
from keras.layers import Dense
from cav import return_split_models

class TestModelSplit(unittest.TestCase):
    ''' Tests for model split function
    '''
    def test_length(self):
        '''
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

if __name__ == '__main__':
    unittest.main()
