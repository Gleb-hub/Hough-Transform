import hough
import argparse
import cv2 as cv
import numpy as np
from os import getenv
from pathlib import Path
import plot

folder_path = Path(getenv('IMAGES_FOLDER_PATH', './images/')).resolve()


def get_img(name):
    img = cv.imread((str(folder_path / name)), cv.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]

    new_h = 2 ** int(np.ceil(np.log2(h)))
    new_w = 2 ** int(np.ceil(np.log2(w)))

    resized_img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    return resized_img


if __name__ == '__main__':
    hough.run(get_img('test.jpg'))
    plot.show()
