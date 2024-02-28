import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


img_path = "test3.jpg"
img = Image.open(img_path)

W = 64


def __img_to_rgb_arrays(img: Image):
    img_data = np.array(img)
    r_array = []
    g_array = []
    b_array = []
    for row in img_data:
        r = []
        g = []
        b = []
        for pixel in row:
            r.append(pixel[0])
            g.append(pixel[1])
            b.append(pixel[2])
        
        r_array.append(r)
        g_array.append(g)
        b_array.append(b)
    return np.array(r_array), np.array(g_array), np.array(b_array)

def calc_sums(img, xmin, xmax):
    res = np.zeros([W, xmax-xmin])
    if xmax - xmin == 1:
        res[:, 0] = img[:, xmin]
    else:
        mid = (xmin + xmax) // 2
        ans1 = calc_sums(img, xmin, mid)
        ans2 = calc_sums(img, mid, xmax)
        for x in range(W):
            for shift in range(xmax-xmin):
                res[x, shift] = ans1[x, shift//2] + ans2[(x + shift//2 + shift%2) % W, shift//2]
    return res


r, g, b = __img_to_rgb_arrays(img)

r = calc_sums(r, 0, W)
g = calc_sums(g, 0, W)
b = calc_sums(b, 0, W)

r_normalized = (r - r.min()) / (r.max() - r.min()) * 255
r_img = Image.fromarray(r_normalized.astype(np.uint8))
r_img.show()

g_normalized = (g - g.min()) / (g.max() - g.min()) * 255
g_img = Image.fromarray(g_normalized.astype(np.uint8))
g_img.show()

b_normalized = (b - b.min()) / (b.max() - b.min()) * 255
b_img = Image.fromarray(b_normalized.astype(np.uint8))
b_img.show()




def fast_hough_transform(image):
    # Реализуйте ваш алгоритм Fast Hough Transform здесь
    pass


def plot_detected_lines(image, detected_lines):
    plt.imshow(image, cmap='gray')
    for rho, theta in detected_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        plt.plot([x1, x2], [y1, y2], 'r')


# Пример использования
image = np.zeros((100, 100))
detected_lines = np.array([[20, np.pi/4], [40, np.pi/3]])  # Пример обнаруженных прямых
plot_detected_lines(image, detected_lines)
plt.show()
