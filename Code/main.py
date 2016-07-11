# -*- coding: utf-8 -*-
"""A program capable of segmenting the upper and lower incisors in panoramic
radiographs.

Example:
    To segment the second upper incisor on radiograph04:

        $ python Code/main.py Data/Radiographs/ 2 4

    See all options with

        $ python Code/main.py --help

"""

import argparse
import os
import numpy as np
import cv2
import landmarks
import radiographs as rg
from auto_init import create_database
from incisor_model import IncisorModel, MODE_FIT_AUTO, MODE_FIT_MANUAL
import Plotter


class FullPaths(argparse.Action):
    """Expand user- and relative-paths.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


def is_dir(dirname):
    """Check if a path is an actual directory.

    Args:
        dirname (str): The path to check.

    Returns:
        bool: True if the given path corresponds with a real directory.

    """
    if not os.path.isdir(dirname):
        msg = "{0} is not a directory".format(dirname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dirname


def get_args():
    """Get CLI arguments and options.
    """
    parser = argparse.ArgumentParser(description="""A program to segment the upper and lower incisors in panoramic radiographs""")

    parser.add_argument('-p', '--preprocess', help="Preprocess the radiographs with an adaptive median filter",
                        action='store_true')
    parser.add_argument('-d', '--database', help="Build an appearance model for the autoinit option.",
                        action='store_true')
    parser.add_argument('-f', '--method', help="The fitting method that should be used",
                        action='store', choices=['auto', 'manual'], default="auto")
    parser.add_argument('-k', '--k', help="Number of pixels either side of point to represent in grey model",
                        action='store', type=int, default=10)
    parser.add_argument('-m', '--m', help="Number of sample points either side of current point for search",
                        action='store', type=int, default=15)
    parser.add_argument('-o', '--out', help="The folder to store the results",
                        action=FullPaths, type=is_dir, default="Plot/")

    parser.add_argument('radiographs', help="The folder with the radiographs",
                        action=FullPaths, type=is_dir)
    parser.add_argument('incisor', help="The index of the incisor to fit [1-8]",
                        action='store', type=int, choices=range(1, 9))
    parser.add_argument('radiograph', help="The index of the radiograph to fit on [1-14]",
                        action='store', type=int, choices=range(1, 15))

    return parser.parse_args()


def main():
    """Read CLI arguments and execute matching action.
    """
    args = get_args()
    if args.preprocess:
        preprocess(args.radiographs)
    if args.database:
        build_database(args.radiographs)

    method = MODE_FIT_AUTO if args.method == 'auto' else MODE_FIT_MANUAL

    fit(args.radiograph, args.incisor, method, args.k, args.m, args.out)


def preprocess(path):
    """Apply an adaptive median filter to the radiographs at the given path and
    stores the results in a separate ``filtered`` folder.

    Args:
        path (str): The location of the dental radiographs.

    """
    images = rg.load(path+"/")
    for ind, img in enumerate(images):
        print "Preprocessing radiographs ["+str(ind+1)+"/14]"
        img = rg.adaptive_median(img, 3, 5)
        cv2.imwrite("%s/filtered/%02d.tif" % (path, ind+1,), img)


def build_database(path):
    """Interactively select ROIs on the dental radiographs for creating the
    appearance model used for the automatic initialization.

    Args:
        path (str): The location of the dental radiographs.

    """
    images = rg.load(path+"/", indices=range(1, 31))
    create_database(images)


def fit(radiograph_index, incisor_ind, fit_mode, k, m, out):
    """Find the region of an incisor, using the given parameters.

    Args:
        radiograph_index (int): The index of the dental radiograph to fit on.
        incisor_ind (int): The index of the incisor to fit.
        fit mode (AUTO|MANUAL): Wheter to ask for a manual initial fit, or try
            to find one automatically.
        k (int): Number of pixels either side of point to represent in grey model.
        m (int): Number of sample points either side of current point for search.
        out (str): The location to store the result.
    """
    # leave-one-out
    train_indices = range(0, 14)
    train_indices.remove(radiograph_index-1)

    lms = landmarks.load_mirrored(incisor_ind)
    test_lm = lms[radiograph_index-1]
    train_lms = [lms[index] for index in train_indices]

    imgs = rg.load()
    test_img = imgs[radiograph_index-1]
    train_imgs = [imgs[index] for index in train_indices]

    # train
    model = IncisorModel(incisor_ind)
    model.train(train_lms, train_imgs, k)

    # fit
    X = model.estimate_fit(test_img, fit_mode)
    X = model.fit(X, test_img, m)

    # evaluate
    ## show live
    Plotter.plot_landmarks_on_image([test_lm, X], test_img, wait=False)
    ## save image with tooth circled
    img = test_img.copy()
    colors = [(255, 0, 0), (0, 255, 0)]
    for ind, lms in enumerate([test_lm, X]):
        points = lms.as_matrix()
        for i in range(len(points) - 1):
            cv2.line(img, (int(points[i, 0]), int(points[i, 1])),
                     (int(points[i + 1, 0]), int(points[i + 1, 1])),
                     colors[ind])
    cv2.imwrite('%s/%02d-%d.png' % (out, radiograph_index, incisor_ind,), img)
    ## save tooth region segmented
    height, width, _ = test_img.shape
    image2 = np.zeros((height, width), np.int8)
    mask = np.array([X.points], dtype=np.int32)
    cv2.fillPoly(image2, [mask], 255)
    maskimage2 = cv2.inRange(image2, 1, 255)
    segmented = cv2.bitwise_and(test_img, test_img, mask=maskimage2)
    cv2.imwrite('%s/%02d-%d-segmented.png' % (out, radiograph_index, incisor_ind,), segmented)


if __name__ == '__main__':
    main()
