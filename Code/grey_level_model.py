# -*- coding: utf-8 -*-
"""Code to create profiles and grey-level models.
"""

import math
import cv2
import numpy as np
from scipy import linspace, asarray
import utils


class GreyLevelModel(object):
    """Represents a statistical model for the grey-level structure in the locality
    of the model points of an incisor.

    We take 2k+1 samples along each profile normal to the model boundary through
    each model point. The value of each sample corresponds to the normalised
    derivative of the grey-level value in the sample point. This is repeated for
    a set of training images. Their mean sample values and covariance matrix give
    a statistical model for the grey-level profile about each model point.

    Attributes:
        profiles: A (n,2k+1) matrix with the mean 2k+1 samples for all n model points.
        covariance: A list of covariance matrices for each model point.
    """

    def __init__(self):
        self.profiles = []
        self.mean_profile = None
        self.covariance = []

    def build(self, images, gimages, models, point_ind, k):
        """Builds the grey-level model.

        Args:
            images: A list of training images (images are considered as ordered).
            models: A list of Landmark instances representing the models of one
                incisor (same order as in image_list).
            k: The number of pixels to sample, from both sides of normal.

        """

        # extracting samples for each image
        for ind in range(len(images)):
            self.profiles.append(Profile(images[ind], gimages[ind], models[ind], point_ind, k))

        # calculate mean and covariance
        mat = []
        for profile in self.profiles:
            mat.append(profile.samples)
        mat = np.array(mat)
        self.mean_profile = (np.mean(mat, axis=0))
        self.covariance = (np.cov(mat, rowvar=0))

    def quality_of_fit(self, samples):
        """Returns the quality of fit of the given profile to this model.

        Args:
            samples: A numpy array with samples.

        """
        return (samples - self.mean_profile).T \
                .dot(self.covariance) \
                .dot(samples - self.mean_profile)

    
class Profile(object):
    """A profile of samples normal to the model boundary through a model point.

    To reduce the effects of global intensity changes we sample the derivative
    along the profile, rather than the absolute grey-level values. We then
    normalise the sample by dividing through the sum of absolute element values.

    Attributes:
        image: The radiograph from which the profile is sampled.
        grad_image: The gradient of the given image.
        model_point ([int, int]): The model point through which the profile is
            sampled. Also the center of the profile.
        points ([(int, int)]): Coordinates of the points in the profile.
        samples ([float]): The normalised derivative along the profile in each
            of the ``points``.
    """

    def __init__(self, image, grad_image, model, point_ind, k):
        """Build a new profile.
        """
        self.image = image
        self.grad_image = grad_image
        self.model_point = model.points[point_ind, :]
        self.k = k
        self.normal = self.__calculate_normal(model.points[(point_ind-1) % 40, :],
                                              model.points[(point_ind+1) % 40, :])
        self.points, self.samples = self.__sample()


    def __calculate_normal(self, p_prev, p_next):
        """Calculates the normal in a model point.

        Args:
            p_prev: The previous model point.
            p_next: The next model point.

        Returns:
            The normal in the given model point.

        """
        n1 = utils.normal(p_prev, self.model_point)
        n2 = utils.normal(self.model_point, p_next)
        n = (n1 + n2) / 2
        return n / np.linalg.norm(n)

    def __sample(self):
        """Take ``num_samples`` in both directions along the normal in a given
        model point.

        Returns:
            The coordinates of points in which the samples are taken and the
            value of each sample.

        """
        # Take a slice of the image in pos and neg normal direction
        pos_points, pos_values, pos_grads = self.__slice_image2(-self.normal)
        neg_points, neg_values, neg_grads = self.__slice_image2(self.normal)

        # Merge the positive and negative slices in one list
        neg_values = neg_values[::-1]  # reverse
        neg_grads = neg_grads[::-1]  # reverse
        neg_points = neg_points[::-1]  # reverse
        points = np.vstack((neg_points, pos_points[1:, :]))
        values = np.append(neg_values, pos_values[1:])
        grads = np.append(neg_grads, pos_grads[1:])

        # Compute the final sample values
        div = max(sum([math.fabs(v) for v in values]), 1)
        samples = [float(g)/div for g in grads]

        return points, samples


    def __slice_image(self, direction, *arg, **kws):
        """Get the coordinates and intensities of ``k`` pixels along a straight
        line, starting in a given point.

        This version uses interpolation, which makes it more accurate, but a lot
        slower too.
        """
        from scipy.ndimage import map_coordinates

        a = asarray(self.model_point)
        b = asarray(self.model_point + direction*self.k)
        coordinates = (a[:, np.newaxis] * linspace(1, 0, self.k+1) +
                       b[:, np.newaxis] * linspace(0, 1, self.k+1))
        values = map_coordinates(self.image, coordinates, order=1, *arg, **kws)
        grad_values = map_coordinates(self.grad_image, coordinates, order=1, *arg, **kws)
        return coordinates.T, values, grad_values


    def __slice_image2(self, direction):
        """Get the coordinates and intensities of ``k`` pixels along a straight
        line, starting in a given point.

        This version doesn't use interpolation, which makes it less accurate, but
        a lot faster too.
        """
        a = asarray(self.model_point)
        b = asarray(self.model_point + direction*self.k)
        coordinates = (a[:, np.newaxis] * linspace(1, 0, self.k+1) +
                       b[:, np.newaxis] * linspace(0, 1, self.k+1))
        values = self.image[coordinates[1].astype(np.int), coordinates[0].astype(np.int)]
        grad_values = self.grad_image[coordinates[1].astype(np.int), coordinates[0].astype(np.int)]
        return coordinates.T, values, grad_values
