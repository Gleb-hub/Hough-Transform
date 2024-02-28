import os
import cv2 as cv
import matplotlib.pyplot as plt

from hough import run 


FOLDER = 'images/'


def get_img(name):
    flow_img = cv.imread(FOLDER + name)
    final_img = cv.imread(FOLDER + name, flow_img.shape[0] * 2)
    plt.imshow(final_img)
    return final_img


images = os.listdir(FOLDER)

img = get_img('test.jpg')


if __name__ == '__main__':
    run(img, plt)
    