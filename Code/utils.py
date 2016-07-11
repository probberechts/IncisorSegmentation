# -*- coding: utf-8 -*-
"""A couple of helper functions, used throughout the rest of the code.
"""

import time
import sys
import numpy as np


class Timer(object):
    """To time the execution of a piece of code.

    Usage:
        with Timer("I do someting"):
            code
    """
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        if self.name:
            print '[%s]' % self.name,

    def __exit__(self, type, value, traceback):
        print 'Elapsed: %s' % (time.time() - self.tstart)


def normal(p1, p2):
    """Computes the normal vector to a line between two points.

    Args:
        p1, p2 : (x1, y1), (x2, y2)

    Returns:
        The normal (-dy, dx)
    """
    return np.array([p1[1] - p2[1], p2[0] - p1[0]])


def medfilt(x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.

    Args:
        x: A 1D numpy array on wich the filter is applied.
        k: The length of the filter.

    Returns:
        The filtered ``x`` array.

    .. _Code from:
        https://gist.github.com/bhawkins/3535131

    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i+1)] = x[j:]
        y[-j:, -(i+1)] = x[-1]
    return np.median(y, axis=1)


def sliding_window(image, search_region, step_size, window_size):
    """Slides a window across a given image.

    Args:
        image: the image that we are going to loop over.
        search_region: the region of the image to search in [(xLT, yLT), (xRB, yRB)]
        step_size: the number of pixels to skip in both the (x,y) direction.
        window_size: (width, height) of the extracted window.

    Yields:
        A tuple containing the x  and y  coordinates of the sliding window,
        along with the window itself.

    """
    for y in range(search_region[0][1], search_region[1][1] - window_size[1], step_size) + \
                [search_region[1][1] - window_size[1]]:
        for x in range(search_region[0][0], search_region[1][0] - window_size[0], step_size) + \
                [search_region[1][0] - window_size[0]]:
            # yield the current window
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def update_progress(progress):
    """Displays or updates a console progress bar

    Args:
        progress: Accepts a float between 0 and 1.
                    Any int will be converted to a float.
                    A value under 0 represents a 'halt'.
                    A value at 1 or bigger represents 100%

    .. _Code from:
        http://stackoverflow.com/a/15860757

    """
    bar_length = 30 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0.0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0.0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1.0
        status = "Done...\r\n"
    block = int(round(bar_length*progress))
    text = "\rPercent: [{0}] {1}% {2}".format("#"*block + "-"*(bar_length-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
