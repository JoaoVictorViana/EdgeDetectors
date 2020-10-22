import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np

erosion_size = 0
max_elem = 2
max_kernel_size = 21


def convolution(image, kernel, morphologic='', average=False, verbose=False):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):

            output[row, col] = np.sum(padded_image[row:row + kernel_row, col:col + kernel_col] * kernel)

            if 'erode' in morphologic:
                output[row, col] = np.min(padded_image[row:row + kernel_row, col:col + kernel_col] - kernel)
            
            if 'dilate' in morphologic:
                output[row, col] = np.max(padded_image[row:row + kernel_row, col:col + kernel_col] - kernel)
            
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()

    return output

