from tensorflow import image as tf_img
from tensorflow import io as tf_io
from PIL import Image
import numpy as np


class ImgProc:


    def preprocess_image(self, image):
        image = tf_img.decode_jpeg(image, channels=3)
        image = tf_img.resize(image, [224, 224])
        image /= 255.0  # normalize to [0,1] range
        return image

    def load_and_preprocess_image(self, path):
        image = tf_io.read_file(path)
        return self.preprocess_image(image)

    def change_range(self, image, label):
        return 2 * image - 1, label

    def open_image(self, name):
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(name)
        image = image.resize((224, 224))
        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array
        return data



if __name__ == '__main__':
    imgProc = ImgProc()
    imgProc.open_image('test.jpg')
