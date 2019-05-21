''' Image segmentation for automated concept discovery '''

import numpy as np

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

class ImageSegmentation():
    ''' Class for creating segmented images

    Attributes:
        tcav: A TCAV object
        discovery_images: A set of discovery images to pull concepts
    '''
    def segment_images(self, n_segments = 10, compactness = 0.1, sigma = 5):
        ''' Function to segment the inputted discovery images

        Args:
            n_segments: An integer specifying the number of segments
            compactness: A number, preferably on log scale, e.g., 0.01, 0.1, 1, 10, 100
            sigma: A number specifying the width of the Gaussian smoothing kernel for preprocessing each image

        Returns:
            something: something
        '''
        return None
