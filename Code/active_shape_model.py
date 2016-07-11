# -*- coding: utf-8 -*-
"""Code for learning an active shape model.
"""

import numpy as np
from procrustes_analysis import GPA
from landmarks import as_vectors


class ASM(object):
    """Class representing an active shape model.

    Attributes:
        mean_shape (Landmarks): The mean shape of this model.
        pc_modes (np.ndarray(n,p)): A numpy (n,p) array representing the PC modes
            with p the number of modes an n/2 the number of shape points.

    """


    def __init__(self, landmarks):
        """Build an active shape model from the landmarks given.

        Args:
            landmarks (Landmarks): The landmark points from which the ASM is learned.

        """

        # Do Generalized Procrustes analysis
        mu, Xnew = GPA(landmarks)

        # covariance calculation
        XnewVec = as_vectors(Xnew)
        S = np.cov(XnewVec, rowvar=0)

        self.k = len(mu.points)      # Number of points
        self.mean_shape = mu
        self.covariance = S
        self.aligned_shapes = Xnew

        # PCA on shapes
        eigvals, eigvecs = np.linalg.eigh(S)
        idx = np.argsort(-eigvals)   # Ensure descending sort
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        self.scores = np.dot(XnewVec, eigvecs)
        self.mean_scores = np.dot(mu.as_vector(), eigvecs)
        self.variance_explained = np.cumsum(eigvals/np.sum(eigvals))

        # Build modes for up to 98% variance
        def index_of_true(arr):
            for index, item in enumerate(arr):
                if item:
                    return index, item
        npcs, _ = index_of_true(self.variance_explained > 0.99)
        npcs += 1

        M = []
        for i in range(0, npcs-1):
            M.append(np.sqrt(eigvals[i]) * eigvecs[:, i])
        self.pc_modes = np.array(M).squeeze().T
