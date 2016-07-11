# -*- coding: utf-8 -*-
"""Code for automatic initialization of the location of an incisor, before the
iterative fitting procedure is started.

Procedure:
----------
1. Locate the jaw split.
2. Build an appearance model for the four upper incisors and one for the four
   lower incisors using PCA.
3. Use a moving window at different scales to detect the region of the four upper
   and the region of the four lower incisors in the enhanced radiograph.
4. Split both found regions in four equally large parts. This results in a
   bounding box for the initial position of each incisor.
5. Position and scale the mean shape of each incisor in its correspoding bounding
   box. Use the grey-level model of each lower incisor to further improve its
   initial position.

"""

import math
import sys
import cv2
import numpy as np

import radiographs as rg
from split_jaws import split_jaws
from grey_level_model import Profile
from utils import sliding_window
import Plotter

# initialize the list of points for the rectangle bbox,
# the temporaray endpoint of the drawing rectangle
# the list of all bounding boxes of selected rois
# and boolean indicating wether drawing of mouse
# is performed or not
bbox = None
rect_endpoint_tmp = []
rect_bbox = []
drawing = False

def create_database(radiographs):
    """Interactively select rectangle ROIs for the four upper/lower incisors
    and store list of bboxes.

    Args:
        radiographs: A list with dental radiographs.

    .. _Based on:
        http://stackoverflow.com/a/36642507
    """
    def draw_rect_roi(event, x, y, flags, param):
        """Mouse callback function"""
        # grab references to the global variables
        global rect_bbox, rect_endpoint_tmp, drawing, bbox

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that drawing is being
        # performed. set rect_endpoint_tmp empty list.
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_endpoint_tmp = []
            rect_bbox = [(x, y)]
            drawing = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # drawing operation is finished
            rect_bbox.append((x, y))
            drawing = False

            # for bbox find upper left and bottom right points
            p_1, p_2 = rect_bbox
            p_1x, p_1y = p_1
            p_2x, p_2y = p_2

            lx = min(p_1x, p_2x)
            ty = min(p_1y, p_2y)
            rx = max(p_1x, p_2x)
            by = max(p_1y, p_2y)

            # add bbox to list if both points are different
            if (lx, ty) != (rx, by):
                bbox = [(lx, ty), (rx, by)]

        # if mouse is drawing set tmp rectangle endpoint to (x,y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_endpoint_tmp = [(x, y)]

    # First do the upper incisors, next the upper incisors
    for is_lower in range(0, 2):
        if is_lower:
            print 'Select the region of the four lower incisors for each radiograph\n', \
                    'and press the c key when done or the d key to ignore the example.'
        else:
            print 'Select the region of the four upper incisors for each radiograph\n', \
                    'and press the c key when done or the d key to ignore the example.'

        bbox_list = []
        for ind, img in enumerate(radiographs):
            if is_lower:
                windowtitle = "Lower incisors [%d/%d]" % (ind+1, len(radiographs),)
            else:
                windowtitle = "Upper incisors [%d/%d]" % (ind+1, len(radiographs),)
            # clone image and setup the mouse callback function
            canvasimg = img.copy()
            # scale image to fit on screen
            canvasimg, scale = rg.resize(canvasimg, 1200, 800)
            cv2.namedWindow(windowtitle)
            cv2.setMouseCallback(windowtitle, draw_rect_roi)

            # keep looping until the 'c' key is pressed
            while True:
                # display the image and wait for a keypress
                rect_cpy = canvasimg.copy()
                if not drawing:
                    if bbox:
                        start_point = bbox[0]
                        end_point_tmp = bbox[1]
                        cv2.rectangle(rect_cpy, start_point, end_point_tmp, (0, 255, 0), 1)
                        cv2.imshow(windowtitle, rect_cpy)
                    else:
                        cv2.imshow(windowtitle, canvasimg)
                elif drawing and rect_endpoint_tmp:
                    start_point = rect_bbox[0]
                    end_point_tmp = rect_endpoint_tmp[0]
                    cv2.rectangle(rect_cpy, start_point, end_point_tmp, (0, 255, 0), 1)
                    cv2.imshow(windowtitle, rect_cpy)

                key = cv2.waitKey(1) & 0xFF
                # if the 'c' key is pressed, break from the loop, and store the example
                if key == ord('c'):
                    bbox_list.append(bbox)
                    break
                # if the 'd' key is pressed, break from the loop and ignore the example
                if key == ord('d'):
                    break
            # close all open windows
            cv2.destroyAllWindows()

        # rescale bounding boxes
        bbox_list = np.array([[(int(p[0]/scale), int(p[1]/scale))
                               for p in bb]
                              for bb in bbox_list])

        # print results summary
        bbs = [bb[1] - bb[0] for bb in bbox_list]
        avg_width, avg_height = np.mean(bbs, axis=0)
        print 'Avg. height: ' + str(avg_height)
        print 'Avg. width: ' + str(avg_width)

        # save the results
        if is_lower:
            np.save('Models/lower_incisor_model', bbox_list)
        else:
            np.save('Models/upper_incisor_model', bbox_list)


def load_database(radiographs, is_upper, rewidth=500, reheight=500):
    """Extracts the ROI's, selected in ``create_database``, from the radiographs
    and scales the image regions to a uniform size.

    Args:
        radiographs: A list with dental radiographs for which a ROI was selected
            in ``create_database``.
        is_upper: whether to load the database for the upper/lower teeth.

    Returns:
        Every image is cropped to the ROI and resized to rewidth*reheight.

    """
    smallImages = np.zeros((14, rewidth * reheight))
    try:
        if is_upper:
            four_incisor_bbox = np.load('Models/upper_incisor_model.npy')
        else:
            four_incisor_bbox = np.load('Models/lower_incisor_model.npy')
    except IOError:
        sys.exit("Create a database first!")

    radiographs = [rg.enhance_fast(radiograph) for radiograph in radiographs]
    for ind, radiograph in enumerate(radiographs):
        [(x1, y1), (x2, y2)] = four_incisor_bbox[ind-1]
        cutImage = radiograph[y1:y2, x1:x2]
        result = cv2.resize(cutImage, (rewidth, reheight), interpolation=cv2.INTER_NEAREST)
        smallImages[ind-1] = result.flatten()

    return smallImages


def project(W, X, mu):
    """Project X on the space spanned by the vectors in W.
    mu is the average image.
    """
    return np.dot(X - mu.T, W)


def reconstruct(W, Y, mu):
    """Reconstruct an image based on its PCA-coefficients Y, the evecs W
    and the average mu.
    """
    return np.dot(Y, W.T) + mu.T


def pca(X, nb_components=0):
    """Do a PCA analysis on X

    Args:
        X: np.array containing the samples
            shape = (nb samples, nb dimensions of each sample)
        nb_components: the nb components we're interested in

    Returns:
        The ``nb_components`` largest evals and evecs of the covariance matrix and
        the average sample.

    """
    [n, d] = X.shape
    if (nb_components <= 0) or (nb_components > n):
        nb_components = n

    mu = np.average(X, axis=0)
    X -= mu.transpose()

    eigenvalues, eigenvectors = np.linalg.eig(np.dot(X, np.transpose(X)))
    eigenvectors = np.dot(np.transpose(X), eigenvectors)

    eig = zip(eigenvalues, np.transpose(eigenvectors))
    eig = map(lambda x: (x[0] * np.linalg.norm(x[1]),
                         x[1] / np.linalg.norm(x[1])), eig)

    eig = sorted(eig, reverse=True, key=lambda x: abs(x[0]))
    eig = eig[:nb_components]

    eigenvalues, eigenvectors = map(np.array, zip(*eig))

    return eigenvalues, np.transpose(eigenvectors), mu


def normalize(img):
    """Normalize an image such that it min=0 , max=255 and type is np.uint8
    """
    return (img*(255./(np.max(img)-np.min(img)))+np.min(img)).astype(np.uint8)


def find_bbox(mean, evecs, image, width, height, is_upper, jaw_split, show=False):
    """Finds a bounding box around the four upper or lower incisors.
    A sliding window is moved over the given image. The window which matches best
    with the given appearance model is returned.

    Args:
        mean: PCA mean.
        evecs: PCA eigen vectors.
        image: The dental radiograph on which the incisors should be located.
        width (int): The default width of the search window.
        height (int): The default height of the search window.
        is_upper (bool): Wheter to look for the upper (True) or lower (False) incisors.
        jaw_split (Path): The jaw split.

    Returns:
        A bounding box around what looks like four incisors.
        The region of the image selected by the bounding box.

    """
    h, w = image.shape

    # [b1, a1]---------------
    # -----------------------
    # -----------------------
    # -----------------------
    # ---------------[b2, a2]

    if is_upper:
        b1 = int(w/2 - w/10)
        b2 = int(w/2 + w/10)
        a1 = int(np.max(jaw_split.get_part(b1, b2), axis=0)[1]) - 350
        a2 = int(np.max(jaw_split.get_part(b1, b2), axis=0)[1])
    else:
        b1 = int(w/2 - w/12)
        b2 = int(w/2 + w/12)
        a1 = int(np.min(jaw_split.get_part(b1, b2), axis=0)[1])
        a2 = int(np.min(jaw_split.get_part(b1, b2), axis=0)[1]) + 350

    search_region = [(b1, a1), (b2, a2)]

    best_score = float("inf")
    best_score_bbox = [(-1, -1), (-1, -1)]
    best_score_img = np.zeros((500, 400))
    for wscale in np.arange(0.8, 1.3, 0.1):
        for hscale in np.arange(0.7, 1.3, 0.1):
            winW = int(width * wscale)
            winH = int(height * hscale)
            for (x, y, window) in sliding_window(image, search_region, step_size=36, window_size=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue

                reCut = cv2.resize(window, (width, height))

                X = reCut.flatten()
                Y = project(evecs, X, mean)
                Xacc = reconstruct(evecs, Y, mean)

                score = np.linalg.norm(Xacc - X)
                if score < best_score:
                    best_score = score
                    best_score_bbox = [(x, y), (x + winW, y + winH)]
                    best_score_img = reCut

                if show:
                    window = [(x, y), (x + winW, y + winH)]
                    Plotter.plot_autoinit(image, window, score, jaw_split, search_region, best_score_bbox,
                                          title="wscale="+str(wscale)+" hscale="+str(hscale))

    return (best_score_bbox, best_score_img)


def fit_template(template, model, img):
    """Try to improve the fit of a shape by trying different configurations for
    position, scale and rotation and returning the configuration with the best
    fit for the grey-level model.

    Args:
        template (Landmarks): The initial fit of an incisor.
        model (GreyLevelModel): The grey-level model of the incisor.
        img: The dental radiograph on which the shape should be fitted.

    Returns:
       Landmarks: The estimated location of the shape.

    """
    gimg = rg.togradient_sobel(img)

    dmin, best = np.inf, None
    for t_x in xrange(-5, 50, 10):
        for t_y in xrange(-50, 50, 10):
            for s in np.arange(0.8, 1.2, 0.1):
                for theta in np.arange(-math.pi/16, math.pi/16, math.pi/16):
                    dists = []
                    X = template.T([t_x, t_y], s, theta)
                    for ind in list(range(15)) + list(range(25,40)):
                        profile = Profile(img, gimg, X, ind, model.k)
                        dist = model.glms[0][ind].quality_of_fit(profile.samples)
                        dists.append(dist)
                    avg_dist = np.mean(np.array(dists))
                    if avg_dist < dmin:
                        dmin = avg_dist
                        best = X

                    Plotter.plot_landmarks_on_image([template, best, X], img, wait=False)

    return best


def init(model, img, show=False):
    """Find an initial estimate for the model in the given image.

    Args:
        model (Landmarks): The shape which should be fitted.
        img: The dental radiograph on which the shape should be fitted.

    Returns:
        Landmarks: An initial estimate for the position of the model in the given image.

    """
    # Are we fitting a lower or an upper incisor?
    tooth = model.incisor_nr
    is_upper = tooth < 5
    if is_upper:
        # UPPER: Avg. height: 314.714285714, Avg. width: 381.380952381
        width = 380
        height = 315
    else:
        # LOWER: Avg. height: 259.518518519, Avg. width: 281.518518519
        width = 280
        height = 260

    # Create the appearance model for the four upper/lower teeth
    radiographs = rg.load()
    data = load_database(radiographs, is_upper, width, height)
    [_, evecs, mean] = pca(data, 5)

    # Visualize the appearance model
    # cv2.imshow('img',np.hstack( (mean.reshape(height,width),
    #                              normalize(evecs[:,0].reshape(height,width)),
    #                              normalize(evecs[:,1].reshape(height,width)),
    #                              normalize(evecs[:,2].reshape(height,width)))
    #                            ).astype(np.uint8))
    # cv2.waitKey(0)

    # Find the jaw split
    jaw_split = split_jaws(img)

    # Find the region of the radiograph that matches best with the appearance model
    img = rg.enhance(img)
    [(a, b), (c, d)], _ = find_bbox(mean, evecs, img, width, height, is_upper, jaw_split, show=show)

    # Assume all teeth have more or less the same width
    ind = tooth if tooth < 5 else tooth - 4
    bbox = [(a +(ind-1)*(c-a)/4, b), (a +(ind)*(c-a)/4, d)]
    center = np.mean(bbox, axis=0)

    # Plot a bounding box around the estimated region of the requested tooth
    if show:
        Plotter.plot_autoinit(img, bbox, 0, jaw_split, wait=True)

    # Position the mean shape of the requested incisor inside the bbox
    template = model.asm.mean_shape.scale_to_bbox(bbox).translate(center)

    # The position of the lower incisors is further improved using the grey-level model
    if is_upper:
        X = template
    else:
        X = template
        # X = fit_template(template, model, img)

    # Show the final result
    if show:
        Plotter.plot_landmarks_on_image([X], img)

    # Return the estimated position of the shape's landmark points
    return X


if __name__ == '__main__':
    # create_database()
    from incisor_model import IncisorModel
    model = IncisorModel.load(5)
    img = rg.load()[7]
    init(model, img, show=False)

