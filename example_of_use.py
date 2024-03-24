from PIL import Image
import numpy as np

img = Image.open("1.png")
arr = 1 - np.asarray(img)[..., 0] / 255
show(arr)
show(create_template(41))
show(hough(arr, 41))