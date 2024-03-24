import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from PIL import Image

def create_template(radius):
    n = round(radius)
    arr = np.zeros((2 * n + 1, 2 * n + 1))
    i = n
    j = 0
    while i >= j:
        for a, b in ((i, j), (j, i)):
            arr[n - a, n - b] = i
            arr[n - a, n + b] = i
            arr[n + a, n - b] = i
            arr[n + a, n + b] = i
        ri = ((i - 1) ** 2 + j ** 2) ** 0.5
        rj = (i ** 2 + (j + 1) ** 2) ** 0.5
        if abs(ri - radius) > abs(rj - radius):
            j += 1
        else:
            i -= 1
    arr /= arr.sum()
    return arr


def hough(array, radius):
    if np.issubdtype(array.dtype, np.integer):
        array = array / 255
    if len(array.shape) == 2:
        return convolve2d(array, create_template(radius), mode = "same")
    conv = np.zeros(array.shape)
    for i in range(array.shape[2]):
        conv[..., i] = convolve2d(array[..., i], create_template(radius), mode = "same")
    return conv


def show(array, figsize = (8, 8)):
    plt.figure(figsize = figsize)
    if len(array.shape) == 2:
        plt.imshow(array, cmap = "gray")
    else:
        plt.imshow(array)
    plt.axis("off")
    plt.show()


def find_circles(image, radius, quantile=0.99):
    '''
    Finds circles of fixed radius in image : np.ndarray | str (path to an image)

    Parameters:
    1) image : np.ndarray | str - grayscale image with circles or path to an image
    2) radius : float - radius of circles in pixels
    3) quantile : float in [0, 1] - quantile of the brightest pixels 
    in the Hough transform to consider them circle centers

    Returns:
    1) list of tuples (x, y) - coordinates of centers of found circles
    '''
    #TODO: Rewrite as you wish
    if isinstance(image, str):
        image = Image.open(image)
        image = np.asarray(image.convert('L'))
    
    transform = hough(image, radius)
    threshold = float(np.quantile(transform.squeeze(), quantile))
    # binarization
    return list(zip(*np.where(transform >= threshold)))