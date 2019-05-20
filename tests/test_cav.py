import unittest
import numpy as np

from context import *
from cav.main import TCAV

class TestTCAV(unittest.TestCase):
    ''' Tests for model split function
    '''
    def setUp(self):
        self.tcav = TCAV()


if __name__ == '__main__':
    unittest.main()
