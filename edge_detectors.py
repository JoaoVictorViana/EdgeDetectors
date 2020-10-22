import cv2
import math
import numpy as np
import scipy as sp
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from convolution import *
from gaussian_smoothing import *
from gradient_operators import *
from utils import *
from convert import *

'''
    Sobel Edge Detector
'''
def sobel_edge_detection(image, filter, verbose=False):
    g_x = convolution(image, filter, morphologic="" ,verbose=False)
 
    if verbose:
        plt.imshow(g_x, cmap='gray')
        plt.title("Horizontal Edge")
        plt.show()
 
    g_y = convolution(image, np.flip(filter.T, axis=0), morphologic="" ,verbose=False)
 
    if verbose:
        plt.imshow(g_y, cmap='gray')
        plt.title("Vertical Edge")
        plt.show()
 
    gradient_magnitude = np.sqrt(np.square(g_x) + np.square(g_y))
 
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
 
    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()
 
    return gradient_magnitude


'''
    Morphologic Edge Detector
'''
def morphologic_edge_detectors(image, kernel, verbose=False):
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

'''
    Laplacian of Gaussian Detector
'''
def LoG(image):
    # LoG
    #blur = cv2.GaussianBlur(image, (5,5), 1)
 
    # Apply Laplacian operator in some higher datatype
    #laplacian = cv2.Laplacian(blur,cv2.CV_64F, 5)
    #laplacian *= 255.0 /laplacian.max()
    image = img_as_float(image)
    laplacian = nd.gaussian_laplace(image , 2)
    thres = np.absolute(laplacian).mean() * 0.90
    output = sp.zeros(laplacian.shape)
    w = output.shape[1]
    h = output.shape[0]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = laplacian[y-1:y+2, x-1:x+2]
            p = laplacian[y, x]
            maxP = patch.max()
            minP = patch.min()
            if (p > 0):
                zeroCross = True if minP < 0 else False
            else:
                zeroCross = True if maxP > 0 else False
            if ((maxP - minP) > thres) and zeroCross:
                output[y, x] = 1
    #z_image = Zero_crossing(laplacian)
    #_,thresh_0 = cv2.threshold(z_image, 20, 255, cv2.THRESH_BINARY)

    return laplacian, output


'''
    Canny Edge Detector
'''
def Canny(image, low, high, sigma=0, with_threshold=False):
    image = Grayscale(GaussianBlur(image, sigma))
    
    if not with_threshold:
        return image

    #image = non_maximum_suppression(image, 0.4)
    grim, gphase = gradient(image)
    image = maximum(grim, gphase)
    #image, angles = SobelFilter(image, sigma=sigma)

    grad = np.copy(image)
    image, _ = thresholding(image, low, high)
    return image, grad
