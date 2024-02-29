import cv2 as cv
from os import getenv
import matplotlib.pyplot as plt

line_thickness = int(getenv('LINE_THICKNESS', 2))
draw_color = tuple(getenv('DRAW_COLOR', (64, 0, 0)))


def draw_img(img, title):
    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    fig.set_figheight(12)
    fig.suptitle(title)
    ax.imshow(img)
    return img


def draw_lines(img, lines, title):
    lines_img = img.copy()
    for points in lines:
        p1, p2 = points
        cv.line(lines_img, p1, p2, draw_color, line_thickness)
    return draw_img(lines_img, title)


def show():
    plt.show()
