import hough
import image
from os import getenv
from pathlib import Path

folder_path = Path(getenv('IMAGES_FOLDER_PATH', './images/')).resolve()

if __name__ == '__main__':
    hough.run(image.get_img(str(folder_path / 'test2.jpg')))
    image.show()
