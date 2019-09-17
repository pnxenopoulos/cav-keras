""" Image segmentation for automated concept discovery """

import numpy as np

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from sklearn.cluster import KMeans


class ImageSegmentation(object):
    """ Class for creating segmented images

    Attributes:
        discovery_images: A set of discovery images to discover concepts from
        image_segments: The discovery images
        activations_model: A Keras Sequential object that is pulled from model_f in the main TCAV object
    """

    def __init__(
        self,
        discovery_images=None,
        activations_model=None,
        n_segments=10,
        compactness=0.1,
    ):
        """ Initialize class
        """
        self.discovery_images = discovery_images
        self.n_segments = n_segments
        self.compactness = compactness
        self.image_segments = None
        self.segmented_images = None
        self.activations_model = activations_model
        self.activations = None

    def set_discovery_images(self, discovery_images=None):
        """ Function to set the discovery images set

        Arg:
            discovery_images: A set of discovery images to discover concepts from
        """
        self.discovery_images = discovery_images

    def set_activations_model(self, activations_model=None):
        """ Function to set the model to find the activations

        Arg:
            activations_model: A Keras Sequential object that is pulled from model_f in the main TCAV object
        """
        self.activations_model = activations_model

    def set_segmentation_options(self, n_segments=10, compactness=0.1):
        """ Function to set image segmentation parameters

        Args:
            n_segments: Number of different images to segment discovery images
            compactness: Balances color proximity and space proximity. Higher values give more weight to space proximity, making superpixel shapes more square/cubic. Try logarithmic values, i.e., 0.01, 0.1, 1, 10, 100
        """
        self.n_segments = n_segments
        self.compactness = compactness

    def segment_images(self, n_segments=10, compactness=0.1, slic_zero=True):
        """ Function to segment the inputted discovery images

        Args:
            n_segments: An integer specifying the number of segments
            compactness: A number, preferably on log scale, e.g., 0.01, 0.1, 1, 10, 100
            sigma: A number specifying the width of the Gaussian smoothing kernel for preprocessing each image

        Returns:
            Segmented images
        """
        if slic_zero:
            self.image_segments = slic(self.discovery_images, slic_zero=True)
        else:
            self.image_segments = slic(
                self.discovery_images,
                n_segments=self.n_segments,
                compactness=self.compactness,
            )

    def resize_segments():
        """ Function to resize the segments
        """
        self.segmented_images = None

    def get_activations():
        """ Function to get the activations of the segmented images
        """
        self.activations = None

    def cluster_activations(self, n_clusters=5):
        """ Function to cluster the activations

        Arg:
            n_clusters: Integer specifying the number of clusters
        """
        kmeans = KMeans(n_clusters=n_clusters).fit(self.activations)
        self.segmented_images_labels = kmeans.labels_
