# -*- coding: utf-8 -*-
"""Some functions to load and manipulate the dental radiographs.
"""

import sys
import os
import hashlib
import cv2
import numpy as np
from scipy.ndimage import morphology
from utils import update_progress


def load(path="Data/Radiographs/", indices=range(1, 15)):
    """Loads a series of radiograph images.

    Args:
        path: the path where the radiographs are stored.
        indices: indices of the radiographs which should be loaded.

    Returns:
        An array with the requested radiographs as 3-channel color images,
        ordered the same as the given indices.
    """

    # load images into images array
    files = ["%02d.tif" % i if i < 15 else "extra/%02d.tif" % i for i in indices]
    files = [path + f for f in files]
    images = [cv2.imread(f) for f in files]

    # check if all loaded files exist
    for index, img in zip(indices, images):
        if img is None:
            raise IOError("%s%02d.tif does not exist" % (path, index,))

    return images


def enhance(img):
    """Enhances a dental x-ray image by
        1. applying a bilateral filter,
        2. combining the top- and bottom-hat transformations
        3. applying CLAHE

    Args:
        img: A dental x-ray image.

    Returns:
        The enhanced radiograph as a grayscale image.

    """
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = adaptive_median(img, 3, 5)
    img = bilateral_filter(img)

    img_top = top_hat_transform(img)
    img_bottom = bottom_hat_transform(img)

    img = cv2.add(img, img_top)
    img = cv2.subtract(img, img_bottom)

    img = clahe(img)

    return img


def enhance_fast(img):
    """Stores the result of enhancing a dental radiograph and reloads the result
    on a next call instead of recomputing it.

    Args:
        img: A grayscale dental x-ray image.

    Returns:
        The enhanced radiograph as a grayscale image.

    """
    directory = "Data/Radiographs/Enhance/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    fname = hashlib.md5(img).hexdigest() + ".png"
    if not os.path.isfile(directory + fname):
        sobel = enhance(img)
        cv2.imwrite(directory + fname, sobel)
        return sobel
    else:
        return cv2.imread(directory + fname, 0)


def clahe(img):
    """Creates a CLAHE object and applies it to the given image.

    Args:
        img: A grayscale dental x-ray image.

    Returns:
        The result of applying CLAHE to the given image.

    """
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    return clahe_obj.apply(img)


def resize(image, width, height):
    """Resizes the given image to the given width and height.

    Args:
        image: The radiograph to resize.
        width (int): The new width for the image.
        height (int): The new height for the image.

    Returns:
        The given image resized to the given width and height, and the scaling factor.

    """
    #find minimum scale to fit image on screen
    scale = min(float(width) / image.shape[1], float(height) / image.shape[0])
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale))), scale


def gauss_pyramid(image, levels):
    """Create a gaussian pyramid of a given image.

    Args:
        image: The source image.
        levels (int): The number of pyramid levels.

    Returns:
        A list of images, the original image as first and the most scaled down one
        as last.

    """
    output = []
    output.append(image)
    tmp = image
    for _ in range(0, levels):
        tmp = cv2.pyrDown(tmp)
        output.append(tmp)
    return output


def togradient_scharr(img):
    """Applies the Scharr Operator.

    Args:
        img: A grayscale dental x-ray image.

    Returns:
        An image with the detected edges bright on a darker background.

    """
    gx = cv2.convertScaleAbs(cv2.Scharr(img, -1, 1, 0))
    gy = cv2.convertScaleAbs(cv2.Scharr(img, -1, 0, 1))
    gradimage = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    gradimage = cv2.equalizeHist(gradimage)
    return gradimage


def togradient_sobel(img):
    """Applies the Sobel Operator.

    Args:
        img: A grayscale dental x-ray image.

    Returns:
        An image with the detected edges bright on a darker background.

    """
    img = cv2.GaussianBlur(img,(3,3),0)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


def togradient_sobel_fast(img):
    """Stores the result of applying the sobel operator to an image and reloads
    the result on a next call instead of recomputing it.

    Args:
        img: A grayscale dental x-ray image.

    Returns:
        An image with the detected edges bright on a darker background.

    """
    directory = "Data/Radiographs/Sobel/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    fname = hashlib.md5(img).hexdigest() + ".png"
    if not os.path.isfile(directory + fname):
        sobel = togradient_sobel(img)
        cv2.imwrite(directory + fname, sobel)
        return sobel
    else:
        return cv2.imread(directory + fname, 0)


def togradient_laplacian(img):
    """Applies the Laplacian Operator.

    Args:
        img: A grayscale dental x-ray image.

    Returns:
        An image with the detected edges bright on a darker background.

    """
    return cv2.Laplacian(img,cv2.CV_64F)


def equalize_histogram(img):
    """Equalizes the histogram of a given image.

    Args:
        img: A grayscale dental x-ray image.

    Returns:
        The equalized version of the input image.

    """
    return cv2.equalizeHist(img)


def adaptive_median(image_array, window, threshold):
    """Applies an adaptive median filter to the image. This is essentially a
    despeckling filter for grayscale images.

    Args:
        image_array: The source image, as a numpy array
        window: Sets the filter window size (must be a scalar between 1 and 5).
                Window size (ws) is defined as W = 2*ws + 1 so that W = 3 is a
                3x3 filter window.
        threshold: Sets the adaptive threshold (0=normal median behavior).
                    Higher values reduce the "aggresiveness" of the filter.

    Returns:
        The filtered image.

    .. _Based on:
        https://github.com/sarnold/adaptive-median/blob/master/adaptive_median.py

    """
    image_array = image_array.copy()
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)


    def med(target_array, array_length):
        """Computes the median of a sublist.
        """
        sorted_array = np.sort(target_array)
        median = sorted_array[array_length/2]
        return median

    # set filter window and image dimensions
    W = 2*window + 1
    ylength, xlength = image_array.shape
    vlength = W*W

    # create 2-D image array and initialize window
    filter_window = np.array(np.zeros((W, W)))
    target_vector = np.array(np.zeros(vlength))
    pixel_count = 0

    try:
        # loop over image with specified window W
        for y in range(window, ylength-(window+1)):
            update_progress(y/float(ylength))
            for x in range(window, xlength-(window+1)):
                # populate window, sort, find median
                filter_window = image_array[y-window:y+window+1, x-window:x+window+1]
                target_vector = np.reshape(filter_window, ((vlength),))
                # internal sort
                median = med(target_vector, vlength)
                # check for threshold
                if not threshold > 0:
                    image_array[y, x] = median
                    pixel_count += 1
                else:
                    scale = np.zeros(vlength)
                    for n in range(vlength):
                        scale[n] = abs(target_vector[n] - median)
                    scale = np.sort(scale)
                    Sk = 1.4826 * (scale[vlength/2])
                    if abs(image_array[y, x] - median) > (threshold * Sk):
                        image_array[y, x] = median
                        pixel_count += 1
        update_progress(1)

    except TypeError:
        print "Error in adaptive median filter function"
        sys.exit(2)

    print pixel_count, "pixel(s) filtered out of", xlength*ylength
    return image_array


def bilateral_filter(img):
    """Applies a bilateral filter to the given image.
    This filter is highly effective in noise removal while keeping edges sharp.

    Args:
        img: A grayscale dental x-ray image.

    Returns:
        The filtered image.

    """
    return cv2.bilateralFilter(img, 9, 175, 175)


def top_hat_transform(img):
    """Calculates the top-hat transformation of a given image.
    This transformation enhances the brighter structures in the image.

    Args:
        img: A grayscale dental x-ray image.

    Returns:
        The top-hat transformation of the input image.

    """
    return morphology.white_tophat(img, size=400)


def bottom_hat_transform(img):
    """Calculates the bottom-hat transformation of a given image.
    This transformation enhances the darker structures in the image.

    Args:
        img: A grayscale dental x-ray image.

    Returns:
        The top-hat transformation of the input image.

    """
    return morphology.black_tophat(img, size=80)
