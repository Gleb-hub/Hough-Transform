import numpy as np
from os import getenv
from helper import find_max
from plot import draw_lines, draw_img

threshold_ratio = float(getenv('THRESHOLD_RATIO', 0.9))


def find_point(n, s, t):
    x = round((n - s) * n / t)
    return x


def fast_hough_transform(sub_img):
    _, n = sub_img.shape

    if n < 2:
        return sub_img[:, 0]

    half = int(n / 2)
    return merge_h(
        fast_hough_transform(sub_img[:, 0:half]),
        fast_hough_transform(sub_img[:, half:n])
    )


def merge_h(h1, h2):
    h, w = h1.shape + (1, ) * (2 - len(h1.shape))

    n0 = w * 2
    r = (w - 1) / (n0 - 1)

    h1, h2 = h1.reshape((h, w)), h2.reshape((h, w))
    result = np.zeros((h, n0))

    for t in range(n0):
        t0 = int(t * r)
        s = t - t0
        result[:, t] = h1[:, t0] + np.concatenate([h2[s: h, t0], h2[0: s, t0]], axis=0)
    return result


def get_line_points(picture, s, t):
    h, w = picture.shape

    if s + t > h:
        x_0 = find_point(h, s, t)

        zero_matrix = np.zeros((h, w))
        stacked_upper = np.vstack([zero_matrix, picture])[s:s + h, :]
        stacked_lower = np.vstack([picture, zero_matrix])[s:s + h, :]

        max_up = find_max(fast_hough_transform(stacked_upper))
        max_low = find_max(fast_hough_transform(stacked_lower))

        if max_up > max_low:
            p1 = (x_0, 0)
            p2 = (w - 1, s + t - h)
        else:
            p1 = (x_0, h)
            p2 = (0, s)
    else:

        p1 = (0, s)
        p2 = (w - 1, s + t)
    return p1, p2


def search_hough_lines(img, threshold_ratio=0.9):
    hough_img = fast_hough_transform(img)
    max_intensity, _ = find_max(hough_img)

    threshold = max_intensity * threshold_ratio
    lines = np.argwhere(hough_img >= threshold)

    lines_points = []
    for s, t in lines:
        lines_points.append(get_line_points(img, s, t))
    return hough_img, lines_points


def run(img):
    hough_img, lines = search_hough_lines(img, threshold_ratio)

    draw_img(img, 'Input Image')
    draw_lines(img, lines, 'Lines Image')
    draw_img(hough_img, 'Hough Image')
