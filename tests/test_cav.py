import unittest
import numpy as np

from context import *
from cav.tcav import TCAV

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam


class TestTCAV(unittest.TestCase):
    """ Tests for model split function
    """

    def setUp(self):
        x_train = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [5, 6, 7, 8, 9]])
        y_train = [1, 1, 0]
        model = Sequential()
        model.add(Dense(5, input_shape=(5,)))
        model.add(Dense(3))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        model.compile(
            loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"]
        )
        model.fit(x_train, y_train, batch_size=1, epochs=1)
        self.model = model
        self.tcav = TCAV(model=model)

    def testModelAssignmentMethod(self):
        """ Test model assignment in TCAV object via .set_model method
        """
        self.tcav = TCAV()
        self.tcav.set_model(self.model)
        self.assertIsInstance(self.tcav.model, Sequential)

    def testModelAssignmentDirect(self):
        """ Test model assignment in TCAV object via object creation
        """
        self.tcav = TCAV(model=self.model)
        self.assertIsInstance(self.tcav.model, Sequential)

    def testSplitModel(self):
        """ Test model splitting functionality for basic dense model
        """
        self.tcav.split_model(bottleneck=1, conv_layer=False)
        self.assertIsInstance(self.tcav.model_f, Sequential)
        self.assertIsInstance(self.tcav.model_h, Sequential)

    def testSplitModelErrors(self):
        """ Test model splitting errors
        """
        with self.assertRaises(ValueError):
            self.tcav.split_model(bottleneck=-1, conv_layer=False)


if __name__ == "__main__":
    unittest.main()
