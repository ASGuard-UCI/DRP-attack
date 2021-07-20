import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os
import numpy as np
import cv2
from keras.layers.core import Lambda
from keras.layers.merge import Average, add
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, Callback
import keras.regularizers as regs
from tqdm import tqdm
ROI_MAT = np.array(
    [
        [0.990908, 0.0, 339.854187],
        [-0.006753, 1.0, 332.119049],
        [-0.000015, 0.0, 1.018190],
    ]
)  # RAV4
ROI_MAT_INV = np.linalg.inv(ROI_MAT)
IMG_CROP_HEIGHT = 256
IMG_CROP_WIDTH = 512


MAP_MAGNET = {
    'mnist': {'model': [3, "average", 3], 'path': '/home/takamisato/lab_gpu/aml/magnet_defense/defensive_models_0916_mnist/'},
    'cifar': {'model': [3, 3, 1], 'path': '/home/takamisato/lab_gpu/aml/magnet_defense/defensive_models_0916_cifar/'},
    'param1': {'model': [16, "average", 32, "average", 64], 'path': '/home/takamisato/lab_gpu/aml/magnet_defense/defensive_models_0915/'},
    'param2': {'model': [16, "average", 8, "average", 8, "average", 8], 'path': '/home/takamisato/lab_gpu/aml/magnet_defense/defensive_models_0916/'},
}

class DenoisingAutoEncoder:
    def __init__(self, image_shape,
                 structure,
                 v_noise=0.0,
                 activation="relu",
                 model_dir="./defensive_models/",
                 reg_strength=0.0):
        """
        Denoising autoencoder.
        image_shape: Shape of input image. e.g. 28, 28, 1.
        structure: Structure of autoencoder.
        v_noise: Volume of noise while training.
        activation: What activation function to use.
        model_dir: Where to save / load model from.
        reg_strength: Strength of L2 regularization.
        """
        h, w, c = image_shape
        self.image_shape = image_shape
        self.model_dir = model_dir
        self.v_noise = v_noise

        input_img = Input(shape=self.image_shape)
        x = input_img

        for layer in structure:
            if isinstance(layer, int):
                x = Conv2D(layer, (3, 3), activation=activation, padding="same",
                           activity_regularizer=regs.l2(reg_strength))(x)
            elif layer == "max":
                x = MaxPooling2D((2, 2), padding="same")(x)
            elif layer == "average":
                x = AveragePooling2D((2, 2), padding="same")(x)
            else:
                print(layer, "is not recognized!")
                exit(0)

        for layer in reversed(structure):
            if isinstance(layer, int):
                x = Conv2D(layer, (3, 3), activation=activation, padding="same",
                           activity_regularizer=regs.l2(reg_strength))(x)
            elif layer == "max" or layer == "average":
                x = UpSampling2D((2, 2))(x)

        decoded = Conv2D(c, (3, 3), activation='sigmoid', padding='same',
                         activity_regularizer=regs.l2(reg_strength))(x)
        self.model = Model(input_img, decoded)

    def train(self, list_img_path, archive_name, num_epochs=100, batch_size=64,
              if_save=True):
        self.model.compile(loss='mean_squared_error',
                           metrics=['mean_squared_error'],
                           optimizer='adam')
        """
        noise = self.v_noise * np.random.normal(size=np.shape(data.train_data))
        noisy_train_data = data.train_data + noise
        noisy_train_data = np.clip(noisy_train_data, 0.0, 1.0)

        self.model.fit(noisy_train_data, data.train_data,
                       batch_size=batch_size,
                       validation_data=(data.validation_data, data.validation_data),
                       epochs=num_epochs,
                       shuffle=True)
        """
        metric = 'val_loss'
        callbacks = [EarlyStopping(monitor=metric,
                                   patience=3,
                                   verbose=1,
                                   min_delta=1e-4,
                                   mode='min'),
                     ReduceLROnPlateau(monitor=metric,
                                       factor=0.1,
                                       patience=4,
                                       verbose=1,
                                       epsilon=1e-4,
                                       mode='min'),
                     ModelCheckpoint(monitor=metric,
                                     filepath=self.model_dir + 'best_weights.hdf5',
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min'),
                     # TensorBoard(log_dir='logs'),
                     ]

        train_size = int(len(list_img_path) * 0.8)
        self.model.fit_generator(generator=data_generator(list_img_path[:train_size], batch_size),
                                 steps_per_epoch=1000,  # np.ceil(float(len(ids_train_split)) / float(batch_size)),
                                 epochs=num_epochs,
                                 verbose=1,
                                 callbacks=callbacks,
                                 validation_data=data_generator(list_img_path[train_size:], batch_size),
                                 validation_steps=len(list_img_path[train_size:])
                                 )
        if if_save:
            self.model.save(os.path.join(self.model_dir, archive_name))

    def load(self, archive_name, model_dir=None):
        if model_dir is None:
            model_dir = self.model_dir
        self.model.load_weights(os.path.join(model_dir, archive_name))
