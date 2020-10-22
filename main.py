import cv2
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from convolution import convolution
from utils import *
from gaussian_smoothing import *
from gradient_operators import *
from edge_detectors import *

'''
    Sobel Edge Detector Example
'''
def example_1(image, filters, gaussian_blur=False, verbose=False):
    image = Grayscale(image)
    
    if gaussian_blur:
        image = GaussianBlur(image, 5)

    images = []
    for filter in filters:
        images.append(sobel_edge_detection(image, filter, verbose=False))
    
    if verbose:
        length = len(filters)
        plot(images, int(length/2), int(length/(length/2)) )

'''
    Canny Edge Detector Example
'''
def example_2(image, low, high, verbose=False):
    images = []
    images.append(Canny(image, low, high, 1))
    images.append(Canny(image, low, high, 2))


    if verbose:
        plot(images, 1, 2)

'''
    Morphologic Edge Detector Example
'''
def example_3(image, rad, verbose=False):
    image = Grayscale(GaussianBlur(image, 2))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    images = []
    images.append(morphologic_edge_detectors(image, kernel1, verbose=False))
    images.append(morphologic_edge_detectors(image, kernel2, verbose=False))

    if verbose:
        plot(images, 1, 2)

'''
    Threshold Example
'''
def example_4(image, threshold, verbose=False):
    image = Grayscale(GaussianBlur(image, 1))

    edges = sobel_edge_detection(image, np.array(SOBEL_OPERATOR), verbose=False)
    _, thresh = cv2.threshold(edges, 20, 255, 0)

    ret, image = cv2.threshold(image, 127, 255, 0)
    img1 = image.copy()

    # Structuring Element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    # Create an empty output image to hold values
    thin = np.zeros(image.shape,dtype='uint8')
    
    # Loop until erosion leads to an empty set
    while (cv2.countNonZero(img1)!=0):
        # Erosion
        # Opening on eroded image
        erode = cv2.erode(img1,kernel)
        opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv2.bitwise_or(subset,thin)
        # Set the eroded image for next iteration
        img1 = erode.copy()

    # aplicar o threshold
    images = []
    images.append(thresh)
    images.append(thin)
    
    if verbose:
        plot(images, 1, 2)

'''
    NMS with Canny Edge Detector Example
'''
def example_5(image, verbose=False):
    image = Grayscale(image)

    image1, _ = Canny(image, 0.06, 0.14, 0.6, with_threshold=True)
    image2, gradient = Canny(image, 0.06, 0.14, 1, with_threshold=True)

    images = []
    images.append(image1)
    images.append(image2)

    if verbose:
        plot(images, 1, 2)

'''
    Zero-Crossing Edge Detector Example
'''
def example_6(image, verbose=False):
    image = Grayscale(GaussianBlur(image, 1))
    
    result = LoG(image)

    if verbose:
        plot(result, 1, 2)


def example_7_8(image1, image2, verbose=False):
    image1 = Grayscale(image1)
    image2 = Grayscale(image2)

    image1, _ = Canny(image1, 0.2, 0.9, 1, with_threshold=True)
    image2, gradient = Canny(image2, 0.2, 0.9, 1, with_threshold=True)

    images1 = []
    images2 = []
    images1.append(image1)
    images1.append(abs(image1 - 255))

    images2.append(image2)
    images2.append(abs(image2 - 255))

    if verbose:
        plot(images1, 1, 2)
        plot(images2, 1, 2)

'''
    New Trends in Edge Detector Example
'''
def example_9_10(image, verbose=False):
    pass

if __name__ == '__main__':
    filters = np.array([
        np.array(PREWITT_OPERATOR),
        np.array(SOBEL_OPERATOR),
        np.array(ROBISON_OPERATOR),
        np.array(KIRSCH_OPERATOR),
        np.array(LAPRACE_L4_OPERATOR),
        np.array(LAPRACE_L8_OPERATOR)
    ])

    image = cv2.imread("images/lena.png")
    image1 = cv2.imread("images/test/42049.jpg")
    image2 = cv2.imread("images/test/253027.jpg")

    # example_1(image, filters, False, True) 
    # example_2(image, 0.06, 0.14, True)
    # example_3(image, 3, True)
    # example_4(image, 0, verbose=True)
    # example_5(image, verbose=True)
    # example_6(image, verbose=True)
    # example_7_8(image1, image2, verbose=True)