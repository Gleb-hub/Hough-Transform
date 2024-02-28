import numpy as np
import cv2 as cv

from helper import find_max


def find_point(n, s, t):
    x = round((n - s) * n / t)
    return x


def fast_hough_transform(picture):
    _, n = picture.shape
    
    if n < 2:
        return picture[:, 0]
    else:
        return merge_h(fast_hough_transform(picture[:, 0: int(n / 2)]), fast_hough_transform(picture[:, int(n / 2): n]))


def merge_h(h1, h2):       
    height = h1.shape[0]
    width = 1 if len(h1.shape) == 1 else h1.shape[1]
    n0 = width * 2
    r = (width - 1) / (n0 - 1)
    h1, h2 = h1.reshape((height, width)), h2.reshape((height, width))
    result = np.zeros((height, n0))
    for t in range(n0):
        t0 = int(t * r)
        s = t - t0
        result[:, t] = h1[:, t0] + np.concatenate([h2[s: height, t0], h2[0: s, t0]], axis=0)
    return result


def draw_line(picture, s, t, plt):
    image = picture.copy()
    height, width = picture.shape
    lineThickness = 2
    if s + t > height:
        x_0 = find_point(height, s, t)
        zero_matrix = np.zeros((height, width))
        stacked_upper = np.vstack([zero_matrix, picture])[s:s+height, :]
        stacked_lower = np.vstack([picture, zero_matrix])[s:s+height, :]
        max_up, = find_max(fast_hough_transform(stacked_upper)), 
        max_low = find_max(fast_hough_transform(stacked_lower))
        if max_up > max_low:
            cv.line(image, (x_0, 0), (width-1, s + t - height), (64, 0, 0), lineThickness)
        else:
            cv.line(image, (x_0, height), (0, s), (64, 0, 0), lineThickness)
        plt.imshow(image)

    else:
        plt.imshow(cv.line(image, (0, s), (width - 1, s + t), (64, 0, 0), lineThickness))
    return image


def ht_PGP_up_and_find_max(img, plt, draw=0):
    obraz = fast_hough_transform(img)
    rez1, (s, t) = find_max(obraz)

    if draw != 0:
        img_with_line = draw_line(img, s, t, plt)
        plt.imshow(img_with_line[::-1,])
    else:
        return rez1, obraz


def run(img, plt):
    max_ = 0

    images = []
    
    max_hft, obr = ht_PGP_up_and_find_max(img, plt)
    images.append(obr)
    if max_hft > max_:
        max_, obr = ht_PGP_up_and_find_max(img, plt)
    
    fig, axes = plt.subplots(2, 2)
    
    for i in range(1):
        axes[i // 2][i % 2].imshow(images[i])


    fig.set_figwidth(12)    
    fig.set_figheight(12)
    fig.suptitle("Hough Images")
    
    ht_PGP_up_and_find_max(img, plt, draw=1)
    plt.show()
