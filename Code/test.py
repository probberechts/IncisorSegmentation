# -*- coding: utf-8 -*-
"""Several functions to test each aspect of the code.
"""

import math
import cv2
import numpy as np

import landmarks
from landmarks import Landmarks
import radiographs as rg
import procrustes_analysis
import auto_init as ai
from active_shape_model import ASM
from grey_level_model import Profile, GreyLevelModel
from incisor_model import IncisorModel, MODE_FIT_MANUAL, MODE_FIT_AUTO
import Plotter


def load_landmarks_and_plot(example_nr=1, incisor_nr=1):
    lm = landmarks.load(incisor_nr)[example_nr-1]
    Plotter.plot_landmarks(lm)


def load_landmarks_and_plot_on_image(examples=range(1, 15), save=False):
    for nr in examples:
        lms = landmarks.load_all_incisors_of_example(nr)
        img = cv2.imread('Data/Radiographs/'+str(nr).zfill(2)+'.tif')
        Plotter.plot_landmarks_on_image(lms, img, save=save, title='Radiograph%02d' % (nr,))


def rotate_landmarks():
    lm = Landmarks('Data/Landmarks/original/landmarks1-1.txt')
    Plotter.plot_landmarks(lm)
    rotated = lm.rotate(math.pi/2)
    Plotter.plot_landmarks(rotated)


def align_shapes():
    X = Landmarks('Data/Landmarks/original/landmarks1-1.txt')
    Y = X.translate([-1, 6])
    Y = Y.rotate(math.pi)
    Y = Y.scale(1.2)
    Plotter.plot_landmarks([X, Y])
    t, s, theta = procrustes_analysis.align_params(X, Y)
    print "Translation: [" + ", ".join(str(f) for f in t) + "] should be [-1, 6]"
    print "Rotation: " + str(theta) + " should be pi"
    print "Scale: " + str(s) + " should be 1.2"


def align_shapes_T():
    X = Landmarks('Data/Landmarks/original/landmarks1-1.txt')
    Y = X.T(np.asarray([-12, -12]), 1.1, -math.pi/2)
    Plotter.plot_landmarks([X, Y])
    t, s, theta = procrustes_analysis.align_params(X, Y)
    print "Translation: [" + ", ".join(str(f) for f in t) + "] should be [-12, -12]"
    print "Rotation: " + str(theta) + " should be 1.6"
    print "Scale: " + str(s) + " should be 1.1"
    Z = Y.invT(np.asarray(t), s, theta)
    Plotter.plot_landmarks([X, Z])


def procrustes():
    ind = 1
    # lms = landmarks.load(ind)
    lms = landmarks.load_mirrored(ind)
    mean_shape, aligned_shapes = procrustes_analysis.GPA(lms)
    Plotter.plot_procrustes(mean_shape, aligned_shapes, incisor_nr=ind, save=True)


def build_asm():
    inc = 1
    lms = landmarks.load_mirrored(inc)
    asm = ASM(lms)
    import pdb; pdb.set_trace()
    Plotter.plot_asm(asm, incisor_nr=inc, save=True)


def create_profile():
    lm = Landmarks('Data/Landmarks/original/landmarks1-1.txt')
    img = rg.load()
    img = rg.enhance(img[0])
    grad_img = rg.togradient_sobel(img)
    profile = Profile(img, grad_img, lm, 30, 40)
    Plotter.plot_profile(grad_img, profile)


def build_grey_level_model():
    lms = landmarks.load(2)
    images = rg.load()
    images = [rg.enhance(img) for img in images]
    gimages = [rg.togradient_sobel(img) for img in images]

    glm = GreyLevelModel()
    glm.build(images, gimages, lms, 10, 20)
    Plotter.plot_grey_level_model(glm, gimages)


def AM_filter():
    img = rg.load()[0]
    filt_img = rg.adaptive_median(img, 3, 5)
    cv2.imwrite('Data/Radiographs/filtered/01.tif', filt_img)


def preprocess():
    img = rg.load()[0]
    img = rg.enhance(img)
    grad = rg.togradient_sobel(img)
    lms = landmarks.load_all_incisors_of_example(1)
    Plotter.plot_landmarks_on_image(lms, grad)


def clean_image():
    img = cv2.imread('Data/Radiographs/test.tif', 1)
    Plotter.plot_image(img, title="original", save=True)

    img = rg.adaptive_median(img, 3, 5)
    Plotter.plot_image(img, title="Adaptive median filtered", save=True)

    img = rg.bilateral_filter(img)
    Plotter.plot_image(img, title="Bilateral filtered", save=True)

    img_top = rg.top_hat_transform(img)
    Plotter.plot_image(img_top, title="Top-hat", save=True)
    img_bottom = rg.bottom_hat_transform(img)
    Plotter.plot_image(img_bottom, title="Bottom-hat", save=True)

    img = cv2.add(img, img_top)
    img = cv2.subtract(img, img_bottom)
    Plotter.plot_image(img, title="Top-hat and bottom-hat combined", save=True)

    img = rg.clahe(img)
    Plotter.plot_image(img, title="CLAHE", save=True)

    grad = rg.togradient_sobel(img)
    Plotter.plot_image(grad, title="Sobel gradient", save=True)


def train_and_save():
    # parameters
    k = 10

    imgs = rg.load()
    for tooth in range(1, 9):
        lms = landmarks.load_mirrored(tooth)
        model = IncisorModel(tooth)
        model.train(lms, imgs, k)
        model.save()


def load_model():
    model = IncisorModel.load(1)
    Plotter.plot_asm(model.asm)


def auto_init():
    imgs = rg.load()
    for radiograph in range(0, 14):
        img = imgs[radiograph]
        X = []
        for tooth in range(1, 9):
            model = IncisorModel.load(tooth)
            X.append(ai.init(model, img))
        Plotter.plot_landmarks_on_image(X, img, show=False, save=True,
                                        title='Autoinit/%02d' % (radiograph,))


def auto_fit():
    # parameters
    m = 15

    imgs = rg.load()
    for radiograph in range(0, 14):
        img = imgs[radiograph]
        Xs = []
        for tooth in range(1, 9):
            model = IncisorModel.load(tooth)
            X = model.estimate_fit(img, MODE_FIT_AUTO)
            X = model.fit(X, img, m)
            Xs.append(X)
        Plotter.plot_landmarks_on_image(Xs, img, show=False, save=True,
                                        title='Autofit/%02d' % (radiograph,))


def fit_model():
    incisor = 8
    m = 15
    model = IncisorModel.load(incisor)
    img = cv2.imread('Data/Radiographs/11.tif')
    X = model.estimate_fit(img, MODE_FIT_AUTO)
    X = model.fit(X, img, m)


if __name__ == '__main__':
    # load_landmarks_and_plot()
    # load_landmarks_and_plot_on_image()
    # rotate_landmarks()
    # align_shapes()
    # align_shapes_T()
    # procrustes()
    build_asm()
    # create_profile()
    # build_grey_level_model()
    # AM_filter()
    # preprocess()
    # clean_image()
    # train_and_save()
    # load_model()
    # auto_init()
    # auto_fit()
    # fit_model()
