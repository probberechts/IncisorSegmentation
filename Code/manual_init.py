# -*- coding: utf-8 -*-
"""Code for manual initialization of the location of an incisor, before the
iterative fitting procedure is started.

The ``init`` procedure allows the user to drag the mean shape of an incisor
to the correct position on a dental radiograph. Manual scaling or rotation
of the mean shape are not supported.
"""

import cv2.cv as cv
import cv2
import numpy as np
import radiographs as rg
from landmarks import Landmarks


tooth = []
tmpTooth = []
dragging = False
start_point = (0, 0)


def init(landmarks, img):
    """Allows the user to provide an initial fit for the given model in the
    given image by dragging the mean shape to the right position on a dental
    radiograph.

    Args:
        landmarks (Landmark): A model.
        img (nparray): An image to fit the model on.

    Returns:
        The centroid of the manual fit.
        The landmark points, adapted to the position chosen by the user and the
        scale of the image.

    """
    global tooth

    oimgh = img.shape[0]
    img, scale = rg.resize(img, 1200, 800)
    imgh = img.shape[0]
    canvasimg = np.array(img)

    # transform model points to image coord
    points = landmarks.as_matrix()
    min_x = abs(points[:, 0].min())
    min_y = abs(points[:, 1].min())
    points = [((point[0]+min_x)*scale, (point[1]+min_y)*scale) for point in points]
    tooth = points
    pimg = np.array([(int(p[0]*imgh), int(p[1]*imgh)) for p in points])
    cv2.polylines(img, [pimg], True, (0, 255, 0))

    # show gui
    cv2.imshow('choose', img)
    cv.SetMouseCallback('choose', __mouse, canvasimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    centroid = np.mean(tooth, axis=0)

    return centroid, Landmarks(np.array([[point[0]*oimgh, point[1]*oimgh] for point in tooth]))


def __mouse(ev, x, y, flags, img):
    """This method handles the mouse-dragging.
    """
    global tooth
    global dragging
    global start_point

    if ev == cv.CV_EVENT_LBUTTONDOWN:
        dragging = True
        start_point = (x, y)
    elif ev == cv.CV_EVENT_LBUTTONUP:
        tooth = tmpTooth
        dragging = False
    elif ev == cv.CV_EVENT_MOUSEMOVE:
        if dragging and tooth != []:
            __move(x, y, img)


def __move(x, y, img):
    """Redraws the incisor on the radiograph while dragging.
    """
    global tmpTooth
    imgh = img.shape[0]
    tmp = np.array(img)
    dx = (x-start_point[0])/float(imgh)
    dy = (y-start_point[1])/float(imgh)

    points = [(p[0]+dx, p[1]+dy) for p in tooth]
    tmpTooth = points

    pimg = np.array([(int(p[0]*imgh), int(p[1]*imgh)) for p in points])
    cv2.polylines(tmp, [pimg], True, (0, 255, 0))
    cv2.imshow('choose', tmp)
