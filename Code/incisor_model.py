# -*- coding: utf-8 -*-
"""Code to create, load and manipulate the active shape model and grey-level
model of one incisor.
"""

import math
import cPickle as pickle
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np

from utils import Timer, medfilt
from active_shape_model import ASM
from grey_level_model import GreyLevelModel
import radiographs as rg
from procrustes_analysis import align_params
from landmarks import Landmarks
from grey_level_model import Profile
import manual_init
import auto_init
import Plotter


# FITTING MODES
MODE_FIT_MANUAL = 0
MODE_FIT_AUTO = 1

# PARAMETERS
PYRAMID_LEVELS = 1
MAX_ITER = 50


class IncisorModel(object):
    """Class representing the model for one incisor.

    Attributes:
        incisor_nr (int): the number of this incisor [1,8]
        asm (ASM): the active shape model for the incisor.
        glm (GreyLevelModel): the grey level models for each landmark point of the incisor.

    """

    def __init__(self, incisor_nr):
        self.incisor_nr = incisor_nr

    def train(self, lms, imgs, k):
        """Learn the active shape model and the grey-level models for different
        levels of a gaussian image pyramid.

        Args:
            lms ([Landmarks]): A training set with landmark points of examples.
            imgs: The corresponding dental radiographs from which the models are learned.
            k (int): Number of pixels either side of point to represent in grey-model.

        """
        self.k = k

        with Timer("Building active shape model"):
            self.asm = ASM(lms)

        with Timer("Building Gaussian image pyramids"):
            pyramids = [rg.gauss_pyramid(image, PYRAMID_LEVELS) for image in imgs]
            lms_pyramids = [[lm.scaleposition(1.0/2**i)
                             for i in range(0, PYRAMID_LEVELS+1)]
                            for lm in lms]

        with Timer("Building grey level models"):
            self.glms = []
            for i in range(0, PYRAMID_LEVELS+1):
                images = [rg.enhance_fast(image) for image in zip(*pyramids)[i]]
                gimages = [rg.togradient_sobel_fast(img) for img in images]
                lms = zip(*lms_pyramids)[i]

                glms = []
                for i in range(0, 40):
                    glm = GreyLevelModel()
                    glm.build(images, gimages, lms, i, k)
                    glms.append(glm)
                self.glms.append(glms)

    def save(self):
        """Save this object.
        """
        f = file('Models/'+str(self.incisor_nr)+'.model', 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def load(incisor_nr):
        """Load the model of a given incisor.

        Args:
            incisor_ind (int): The index of the incisor for which the models
            should be loaded. [1-8]

        Returns:
            IncisorModel: The model of the requsted incisor.

        """
        with file('Models/'+str(incisor_nr)+'.model', 'rb') as f:
            return pickle.load(f)

    def estimate_fit(self, testimg, mode):
        """Find an initial estimate for the model in the given image. This can
        be done manually or automatically.

        Args:
            testimg: A dental radiograph on which the model should be fitted.
            mode: MODE_FIT_AUTO or MODE_FIT_MANUAL

        Returns:
            Landmarks: An initial estimate for the location of the model.

        """
        if mode == MODE_FIT_MANUAL:
            _, X = manual_init.init(self.asm.mean_shape, testimg)
        elif mode == MODE_FIT_AUTO:
            with Timer("Automatically initialize fit"):
                X = auto_init.init(self, testimg)
        return X

    def fit(self, X, testimg, m):
        """Iteratively improve the fit of the model on the given image.

        The search is implemented in a multi-resolution framework, using a gaussian
        image pyramid.

        Args:
            X (Landmarks): An initial estimate for the location of the model.
            testimg: A dental radiograph on which the model should be fitted.
            m (int): Number of sample points either side of current point when
                searching for a better fit of a landmark point.

        Returns:
            Landmarks: The fitted shape.

        .. _Based on:
            An introduction to Active Shape Models - Cootes et al.

        """
        if not m > self.k:
            raise ValueError("m <= k")

        with Timer("Iteratively improving the fit"):
            pyramid = rg.gauss_pyramid(testimg, PYRAMID_LEVELS)
            X = X.scaleposition(1.0 / 2**(PYRAMID_LEVELS+1))
            for img, glms in zip(reversed(pyramid), reversed(self.glms)):
                X = X.scaleposition(2)
                X = self.__fit_one_level(X, img, glms, m, MAX_ITER)
        return X


    def __fit_one_level(self, X, testimg, glms, m, max_iter):
        """Fit the model for one level of the image pyramid.
        """
        # Prepare test image
        img = rg.enhance(testimg)
        gimg = rg.togradient_sobel(img)

        # 0. Initialise the shape parameters, b, to zero (the mean shape)
        b = np.zeros(self.asm.pc_modes.shape[1])
        X_prev = Landmarks(np.zeros_like(X.points))

        # 4. Repeat until convergence.
        nb_iter = 0
        n_close = 0
        best = np.inf
        best_Y = None
        total_s = 1
        total_theta = 0
        while (n_close < 16 and nb_iter <= max_iter):

            with Timer("Fit iteration %d" % (nb_iter,)):
                # 1. Examine a region of the image around each point Xi to find the
                # best nearby match for the point
                Y, n_close, quality = self.__findfits(X, img, gimg, glms, m)
                if quality < best:
                    best = quality
                    best_Y = Y
                Plotter.plot_landmarks_on_image([X, Y], testimg, wait=False,
                                                title="Fitting incisor nr. %d" % (self.incisor_nr,))

                # no good fit found => go back to best one
                if nb_iter == max_iter:
                    Y = best_Y

                # 2. Update the parameters (Xt, Yt, s, theta, b) to best fit the
                # new found points X
                b, t, s, theta = self.__update_fit_params(X, Y, testimg)

                # 3. Apply constraints to the parameters, b, to ensure plausible
                # shapes
                # We clip each element b_i of b to b_max*sqrt(l_i) where l_i is the
                # corresponding eigenvalue.
                b = np.clip(b, -3, 3)
                # t = np.clip(t, -5, 5)
                # limit scaling
                s = np.clip(s, 0.95, 1.05)
                if total_s * s > 1.20 or total_s * s < 0.8:
                    s = 1
                total_s *= s
                # limit rotation
                theta = np.clip(theta, -math.pi/8, math.pi/8)
                if total_theta + theta > math.pi/4 or total_theta + theta < - math.pi/4:
                    theta = 0
                total_theta += theta

                # The positions of the model points in the image, X, are then given
                # by X = TXt,Yt,s,theta(X + Pb)
                X_prev = X
                X = Landmarks(X.as_vector() + np.dot(self.asm.pc_modes, b)).T(t, s, theta)
                Plotter.plot_landmarks_on_image([X_prev, X], testimg, wait=False,
                                                title="Fitting incisor nr. %d" % (self.incisor_nr,))

                nb_iter += 1

        return X

    def __findfits(self, X, img, gimg, glms, m):
        """Examines a region of the given image around each point X_i to find
        best nearby match Y_i.

        Args:
            X (Landmarks): A landmarks model.
            img (np.ndarray): The radiograph from which X was extracted.
            glms (List[GreyLevelModel]): The grey-level models of this incisor.
            m (int): The number of samples used to find a fit.

        Returns:
            Landmarks: A Landmarks object, containing a new fit for each point in X.
            int: the percentage of times that the best found pixel along a search
                    profile is within the center 50% of the profile

        """
        fits = []
        n_close = 0

        profiles = []
        bests = []
        qualities = []
        for ind in range(len(X.points)):
            # 1. Sample a profile m pixels either side of the current point
            profile = Profile(img, gimg, X, ind, m)
            profiles.append(profile)

            # 2. Test the quality of fit of the corresponding grey-level model
            # at each of the 2(m-k)+1 possible positions along the sample
            dmin, best = np.inf, None
            dists = []
            for i in range(self.k, self.k+2*(m-self.k)+1):
                subprofile = profile.samples[i-self.k:i+self.k+1]
                dist = glms[ind].quality_of_fit(subprofile)
                dists.append(dist)
                if dist < dmin:
                    dmin = dist
                    best = i


            # 3. Choose the one which gives the best match
            bests.append(best)
            qualities.append(dmin)
            best_point = [int(c) for c in profile.points[best, :]]

            # 4. Check wheter the best found pixel along the search profile
            # is withing the central 50% of the profile
            is_upper = True if self.incisor_nr < 5 else False
            if (((is_upper and (ind > 9 and ind < 31)) or
                 (not is_upper and (ind < 11 or ind > 29))) and
                    best > 3*m/4 and best < 5*m/4):
                n_close += 1

            # Plotter.plot_fits(gimg, profile, glms[ind], dists, best_point, self.k, m)

        # remove outliers
        bests.extend(bests)
        bests = np.rint(medfilt(np.asarray(bests), 5)).astype(int)
        for best, profile in zip(bests, profiles):
            best_point = [int(c) for c in profile.points[best, :]]
            fits.append(best_point)

        # fit quality
        is_upper = True if self.incisor_nr < 5 else False
        if is_upper:
            quality = np.mean(qualities[10:30])
        else:
            quality = np.mean(qualities[0:10] + qualities[30:40])

        return Landmarks(np.array(fits)), n_close, quality

    def __update_fit_params(self, X, Y, testimg):
        """Finds the best pose (translation, scale and rotation) and shape
        parameters to match a model instance X to a new set of image points Y.

        Args:
            X: a model instance.
            Y: a new set of image points.

        Returns:
            The best pose and shape parameters.

        .. _An Introduction to Active Shape Models:
            Protocol 1 (p9)
        """

        # 1. Initialise the shape parameters, b, to zero (the mean shape).
        b = np.zeros(self.asm.pc_modes.shape[1])
        b_prev = np.ones(self.asm.pc_modes.shape[1])
        i = 0
        while (np.mean(np.abs(b-b_prev)) >= 1e-14):
            i += 1
            # 2. Generate the model point positions using x = X + Pb
            x = Landmarks(X.as_vector() + np.dot(self.asm.pc_modes, b))

            # 3. Find the pose parameters (Xt, Yt, s, theta) which best align the
            # model points x to the current found points Y
            is_upper = True if self.incisor_nr < 5 else False
            t, s, theta = align_params(x.get_crown(is_upper), Y.get_crown(is_upper))

            # 4. Project Y into the model co-ordinate frame by inverting the
            # transformation T
            y = Y.invT(t, s, theta)

            # 5. Project y into the tangent plane to X by scaling:
            # y' = y / (y*X).
            yacc = Landmarks(y.as_vector() / np.dot(y.as_vector(), X.as_vector().T))

            # 6. Update the model parameters to match to y': b = PT(y' - X)
            b_prev = b
            b = np.dot(self.asm.pc_modes.T, (yacc.as_vector()-X.as_vector()))

            # 7. If not converged, return to step 2

        return b, t, s, theta
