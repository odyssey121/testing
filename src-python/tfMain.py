from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import dirname, join,sep
from os import listdir
import pathlib
import tensorflow as tf
from tensorflow.python.data.experimental.ops import shuffle_ops
import random
from tensorflow import keras
import numpy as np
from imgProc import ImgProc
import json


class TfMain:

    BATCH_SIZE = 32
    BATCH_SHUFFLE_SIZE=512
    SHUFFLE_BUFFER_SIZE=2048

    def __init__(self):
        self.imgProc = ImgProc()
        self.DATA_ROOT = pathlib.Path(dirname(__file__))
        self.TRAIN_DIR = pathlib.Path(join(dirname(__file__),'train'))
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        

    def save_labels(self,labelsObj):
        with open(join(self.TRAIN_DIR,'labels.txt'), 'w') as file:
            file.write(json.dumps(labelsObj))

    def load_labels(self):
        labels = None
        with open(join(self.TRAIN_DIR,'labels.txt'), 'r') as file:
            labels = file.read()
        return labels

    def get_paths_labels(self):
        all_image_paths = list(self.DATA_ROOT.glob('./train/*/*'))
        all_image_paths = [str(path) for path in all_image_paths]
        random.shuffle(all_image_paths)

        self.IMAGE_COUNT = len(all_image_paths)

        self.LABEL_NAMES = sorted(
            item.name for item in self.DATA_ROOT.glob('./train/*') if item.is_dir())

        label_to_index = dict((name, index)
                              for index, name in enumerate(self.LABEL_NAMES))
        self.save_labels(label_to_index)
        all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                            for path in all_image_paths]

        return all_image_paths, all_image_labels

    def image_label_ds(self):
        all_image_paths, all_image_labels = self.get_paths_labels()
        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        image_ds = path_ds.map(
            self.imgProc.load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(
            tf.cast(all_image_labels, tf.int64))
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
        return image_label_ds
        

    def preparation_ds(self, image_label_ds):
        ds = image_label_ds.shuffle(buffer_size=self.SHUFFLE_BUFFER_SIZE,reshuffle_each_iteration=True).repeat(-1)
        ds = ds.batch(self.BATCH_SIZE)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds

    def get_model(self, ds=None):
        # простого примера передачи обучения (transfer learning)
        mobile_net = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False)
        mobile_net.trainable = False
        # keras_ds = ds.map(self.imgProc.change_range)
        # image_batch, label_batch = next(iter(keras_ds))
        # feature_map_batch = mobile_net(image_batch)
        model = tf.keras.Sequential([mobile_net, tf.keras.layers.GlobalAveragePooling2D(),
                                     tf.keras.layers.Dense(len(self.LABEL_NAMES), activation="softmax")])
        return model

    def train_model(self, save=True, model_name='custom_model.h5'):
        image_label_ds = self.image_label_ds()
        preparation_ds = self.preparation_ds(image_label_ds)
        model = self.get_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='sparse_categorical_crossentropy',
                      metrics=["accuracy"])
        steps_per_epoch=tf.math.ceil(self.IMAGE_COUNT/self.BATCH_SIZE).numpy()
        model.fit(preparation_ds, epochs=1, steps_per_epoch=80)
        model.save(join(self.DATA_ROOT,model_name))

    def load_model(self, model_name='custom_model.h5'):
        loaded_model = tf.keras.models.load_model(join(self.DATA_ROOT,model_name))
        loaded_model.summary()
        return loaded_model
        # prep_img = self.imgProc.open_image('./src-python/photo/wheat/1ab595d7-98b5-4d10-a4c9-bbf7bed52ad8.jpg')
        # prediction =loaded_model.predict(prep_img)
        # print(np.argmax(prediction[0]))

    def test_prediction(self):
        model = self.load_model()
        labels = json.loads(self.load_labels())
        pathLibItems = [item for item in self.TRAIN_DIR.glob('./*') if item.is_dir()]
        for pathLibItem in pathLibItems:
            currentDir = pathLibItem.absolute()
            for photoName in listdir(currentDir):
                photoPath = join(currentDir,photoName)
                prepImg = self.imgProc.open_image(photoPath)
                prediction = model.predict(prepImg)
                if labels[currentDir.name] != np.argmax(prediction[0]):
                    print(f'error in {currentDir.name} prediction false {photoPath} prediction = {np.argmax(prediction[0])}')

                
                
            









            

        



if __name__ == '__main__':
    tfMain = TfMain()
    tfMain.train_model()

