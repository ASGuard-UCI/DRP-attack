import pickle
import os
from logging import getLogger
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm
from car_motion_attack.models import create_model

from car_motion_attack.load_sensor_data import load_sensor_data
from car_motion_attack.car_motion import CarMotion
from car_motion_attack.config import (DTYPE, PIXELS_PER_METER, SKY_HEIGHT, IMG_INPUT_SHAPE,
                                      IMG_INPUT_MASK_SHAPE, RNN_INPUT_SHAPE,
                                      MODEL_DESIRE_INPUT_SHAPE, MODEL_OUTPUT_SHAPE,
                                      YUV_MIN, YUV_MAX, MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH
                                      )


logger = getLogger(None)


class ReplayBicycle:
    def __init__(
        self,
        sess,
        list_bgr_img,
        df_sensors,
        global_bev_mask,
        roi_mat,
        n_epoch=1001,
        model_path="data/model_keras_20191230_v0.7/driving_model.h5",
        # model_path="data/model_keras_20191125/driving_model.h5",
        learning_rate_patch=1.0e-2,
        learning_rate_color=1.0e-3,
        scale=1,
    ):
        K.set_session(sess)
        self.list_bgr_img = list_bgr_img
        self.n_frames = len(list_bgr_img)
        self.df_sensors = df_sensors
        self.roi_mat = roi_mat

        self.global_bev_mask = global_bev_mask
        self.car_motion = CarMotion(
            self.list_bgr_img,
            self.df_sensors,
            self.global_bev_mask,
            self.roi_mat,
            scale=scale,
        )

        self.global_bev_purtabation = (
            np.ones(
                (self.global_bev_mask.shape[0], self.global_bev_mask.shape[1], 6),
                dtype=DTYPE,
            )
            * 1.0e-10
        )
        self.masked_global_bev_purtabation = self.global_bev_purtabation.copy()

        self.global_base_color = np.array(
            [-0.7, 0, 0], dtype=DTYPE
        )  # np.zeros(3, dtype=DTYPE)

        self.model = create_model()
        self.model.load_weights(model_path)

        self.n_epoch = n_epoch
        self.learning_rate_patch = learning_rate_patch
        self.learning_rate_color = learning_rate_color

    def run(self, lateral_shift=4, starting_meters=60, start_steering_angle=None):
        logger.debug("enter")
        # initialize car model
        self.car_motion.setup_masks(
            lateral_shift=lateral_shift, starting_meters=starting_meters
        )
        # self.list_ops_model_img = self.list_tf_model_imgs

        for _ in range(5):
            logger.debug("calc model ouput")
            model_img_inputs = self.car_motion.calc_model_inputs()

            model_rnn_inputs = []
            model_outputs = []
            rnn_input = np.zeros(RNN_INPUT_SHAPE)
            desire_input = np.zeros(MODEL_DESIRE_INPUT_SHAPE)
            for i in range(self.n_frames):
                model_output = self.model.predict(
                    [
                        model_img_inputs[i].reshape(IMG_INPUT_SHAPE),
                        desire_input,
                        rnn_input,
                    ]
                )
                model_outputs.append(model_output)
                rnn_input = model_output[:, -512:]
                model_rnn_inputs.append(model_output[:, -512:])

            model_outputs = np.vstack(model_outputs)
            self.car_motion.update_trajectory(
                model_outputs, start_steering_angle=start_steering_angle
            )

        #np.save('benign_model_in_1201', model_img_inputs)
        #np.save('benign_model_out_1201', model_outputs)
        #print("AAA", self.car_motion.list_total_lateral_shift)
        logger.debug("exit")
        return self.car_motion
