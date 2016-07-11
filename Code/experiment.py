# -*- coding: utf-8 -*-
"""Evaluate the incisor segmenting algorithm.
"""

from math import sqrt
import cv2
import numpy as np
import landmarks
import radiographs as rg
from incisor_model import IncisorModel, MODE_FIT_AUTO, MODE_FIT_MANUAL
import Plotter


def fit_quality(truth, fit):
    """Computes the average point to associated border distance.

    Args:
        truth: The ground truth lanmark points.
        fit: The fitted shape.

    Returns:
        The average distance between each fitted point and the ground truth.

    """
    dist = []
    for p1, p2 in zip(truth.points, fit.points):
        dist.append(sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2))
    return np.mean(dist)


def fit_quality2(truth, fit):
    """Computes the F-score of the fit.

    F-score = 2*TP / (2*TP + FP + FN)

    Args:
        truth: The ground truth incisor segmentation image.
        fit: The fitted segmentation image.

    Returns:
        The precision of the fit, compared to the ground truth.

    """
    TP = fit & truth
    FP = fit - truth
    FN = truth - fit
    # TN = 255 - (truth + fit)
    return float(2*(TP/255).sum()) / (2*(TP/255).sum() + (FP/255).sum() + (FN/255).sum())


def reset_results():
    """The fit quality of each incisor is stored in a file. This method
    intializes/resets this file.
    """
    results = np.zeros((14, 8))
    np.savetxt('experiment_results.csv', results, fmt='%10.2f', delimiter=',')


def test_fit(test_index, incisor_ind, fit_mode):
    """Test the fit quality of one incisor using leave-one-out analysis.

    Args:
        test_index (int): The index of the radiograph to fit on. [0-13]
        incisor_ind (int): The index of the incisor to fit. [1-8]
        fit_mode: MODE_FIT_AUTO or MODE_FIT_MANUAL

    Returns:
        Landmarks: The fitted model.

    """
    # parameters
    k = 10
    m = 15

    # leave-one-out
    train_indices = range(0, 14)
    train_indices.remove(test_index)

    lms = landmarks.load_mirrored(incisor_ind)
    test_lm = lms[test_index]
    train_lms = [lms[index] for index in train_indices]

    imgs = rg.load()
    test_img = imgs[test_index]
    train_imgs = [imgs[index] for index in train_indices]

    # train
    model = IncisorModel(incisor_ind)
    model.train(train_lms, train_imgs, k)

    # test
    X = model.estimate_fit(test_img, fit_mode)
    X = model.fit(X, test_img, m)

    # evaluate
    Plotter.plot_landmarks_on_image([test_lm, X], test_img, wait=False)
    fit = fit_quality(test_lm, X)

    # save result
    ## landmark annotated
    img = test_img.copy()
    colors = [(255, 0, 0), (0, 255, 0)]
    for ind, lms in enumerate([test_lm, X]):
        points = lms.as_matrix()
        for i in range(len(points) - 1):
            cv2.line(img, (int(points[i, 0]), int(points[i, 1])),
                     (int(points[i + 1, 0]), int(points[i + 1, 1])),
                     colors[ind])
    cv2.imwrite('Plot/Results/%02d-%d.png' % (test_index+1, incisor_ind,), img)

    ## tooth region segmented
    height, width, _ = test_img.shape
    image2 = np.zeros((height, width), np.int8)
    mask = np.array([X.points], dtype=np.int32)
    cv2.fillPoly(image2, [mask], 255)
    maskimage2 = cv2.inRange(image2, 1, 255)
    out = cv2.bitwise_and(test_img, test_img, mask=maskimage2)
    cv2.imwrite('Plot/Results/Segmentations/%02d-%d.png' % (test_index+1, incisor_ind-1,), out)

    truth = cv2.imread('Data/Segmentations/%02d-%d.png' % (test_index+1, incisor_ind-1,), 0)
    (_, truth) = cv2.threshold(truth, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    fit = cv2.imread('Plot/Results/Segmentations/%02d-%d.png' % (test_index+1, incisor_ind-1,), 0)
    (_, fit) = cv2.threshold(fit, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    fit2 = fit_quality2(truth, fit)

    ## fit quality
    results = np.loadtxt('experiment_results.csv', delimiter=',')
    results[test_index, incisor_ind-1] = fit2
    np.savetxt('experiment_results.csv', results, fmt='%10.2f', delimiter=',')

    return X


def run_experiment(mode):
    """Test the fit quality for all training examples.

    Args:
        mode: MODE_FIT_AUTO or MODE_FIT_MANUAL

    """
    imgs = rg.load()
    for radiograph in range(0, 14):
        Xs = []
        for tooth in range(1, 9):
            X = test_fit(radiograph, tooth, mode)
            Xs.append(X)
        img = imgs[radiograph]
        Plotter.plot_landmarks_on_image(Xs, img, show=False, save=True,
                                        title='Results/%02d' % (radiograph+1,))


if __name__ == '__main__':
    reset_results()
    run_experiment(MODE_FIT_MANUAL)
    # test_fit(13, 6, MODE_FIT_AUTO)

