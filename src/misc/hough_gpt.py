import numpy as np
import matplotlib.pyplot as plt

def fast_hough_transform(image, angle_step=1):
    # Получаем размеры изображения
    height, width = image.shape
    
    # Определяем максимальное расстояние rho
    max_dist = int(np.ceil(np.sqrt(height**2 + width**2)))
    
    # Выбираем диапазон углов в радианах
    theta_range = np.deg2rad(np.arange(-30, 30, angle_step))
    
    # Создаем пустое пространство Хафа
    hough_space = np.zeros((2 * max_dist, len(theta_range)))
    
    # Проходим по каждому пикселю изображения
    for y in range(height):
        for x in range(width):
            # Если пиксель является частью линии
            if image[y, x] > 0:
                # Проходим по каждому углу
                for theta_idx, theta in enumerate(theta_range):
                    # Вычисляем rho
                    rho = int(np.round(x * np.cos(theta) + y * np.sin(theta))) + max_dist
                    # Увеличиваем значение в соответствующей ячейке пространства Хафа
                    hough_space[rho, theta_idx] += 1

    return hough_space

def plot_detected_lines(image, hough_space, angle_step=1, threshold=50):
    # Получаем размеры изображения
    height, width = image.shape

    # Определяем максимальное расстояние rho
    max_dist = int(np.ceil(np.sqrt(height**2 + width**2)))

    # Выбираем диапазон углов в радианах
    theta_range = np.deg2rad(np.arange(-30, 30, angle_step))

    # Находим прямые, количество голосов для которых превышает порог
    detected_lines = []
    for rho in range(2 * max_dist):
        for theta_idx in range(len(theta_range)):
            if hough_space[rho, theta_idx] > threshold:
                detected_lines.append((rho - max_dist, theta_range[theta_idx]))

    # Рисуем обнаруженные прямые
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
    plt.show()

# Пример использования
image = np.zeros((100, 100))
image[30:70, 40:60] = 1  # Рисуем прямую на изображении

hough_space = fast_hough_transform(image)
plot_detected_lines(image, hough_space, angle_step=1, threshold=30)
