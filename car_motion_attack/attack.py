import os
import pickle
from logging import getLogger

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tqdm import tqdm

from car_motion_attack.config import MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH
from car_motion_attack.car_motion import CarMotion
from car_motion_attack.load_sensor_data import load_sensor_data
from car_motion_attack.utils import AdamOpt
from car_motion_attack.models import create_model
from car_motion_attack.replay_bicycle import ReplayBicycle
from car_motion_attack.loss import compute_path_pinv, loss_func
from car_motion_attack.config import (DTYPE, PIXELS_PER_METER, SKY_HEIGHT, IMG_INPUT_SHAPE,
                                      IMG_INPUT_MASK_SHAPE, RNN_INPUT_SHAPE,
                                      MODEL_DESIRE_INPUT_SHAPE, MODEL_OUTPUT_SHAPE,
                                      YUV_MIN, YUV_MAX
                                      )

logger = getLogger(None)


class CarMotionAttack:
    def __init__(
        self,
        sess,
        list_bgr_img,
        df_sensors,
        global_bev_mask,
        base_color,
        roi_mat,
        n_epoch=10000,
        model_path="data/model_keras_20191230_v0.7/driving_model.h5",
        # model_path="data/model_keras_20191125/driving_model.h5",
        learning_rate_patch=1.0e-2,
        learning_rate_color=1.0e-3,
        scale=1,
        result_dir='./result/',
        perturbable_area_ratio=50,
        is_attack_to_rigth=True,
        left_lane_pos=4,
        right_lane_pos=36,
        src_corners=None,
        target_deviation=0.5,
        l2_weight=0.01
    ):
        self.sess = sess
        self.list_bgr_img = list_bgr_img
        self.n_frames = len(list_bgr_img)
        self.df_sensors = df_sensors
        self.result_dir = result_dir
        self.perturbable_area_ratio = perturbable_area_ratio
        self.base_color = base_color
        self.roi_mat = roi_mat
        self.is_attack_to_rigth = is_attack_to_rigth
        self.left_lane_pos = left_lane_pos
        self.right_lane_pos = right_lane_pos
        self.target_deviation = target_deviation
        self.l2_weight = l2_weight

        self.last_epoch = None

        self.global_bev_mask = global_bev_mask
        self.car_motion = CarMotion(
            self.list_bgr_img,
            self.df_sensors,
            self.global_bev_mask,
            self.roi_mat,
            left_lane_pos=left_lane_pos,
            right_lane_pos=right_lane_pos,
            scale=scale,
            src_corners=src_corners
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
            [base_color, 0, 0], dtype=DTYPE
        )  # np.zeros(3, dtype=DTYPE)

        self.model = create_model()
        self.model.load_weights(model_path)

        self.n_epoch = n_epoch + 1
        self.learning_rate_patch = learning_rate_patch
        self.learning_rate_color = learning_rate_color

        self._create_tf_variables()

    def _create_tf_variables(self):
        # placeholders
        self.buf_img = tf.placeholder(DTYPE, shape=IMG_INPUT_SHAPE)
        self.buf_rnn_data = tf.placeholder(DTYPE, shape=RNN_INPUT_SHAPE)
        self.buf_model_output = tf.placeholder(DTYPE, shape=MODEL_OUTPUT_SHAPE)
        self.buf_grad = tf.placeholder(DTYPE, shape=IMG_INPUT_SHAPE)
        self.yuv_color = tf.placeholder(DTYPE, shape=3)

        # variables
        self.list_tf_model_patches = [
            tf.Variable(np.zeros(IMG_INPUT_SHAPE, dtype=DTYPE))
            for _ in range(self.n_frames)
        ]

        self.list_tf_model_imgs = [
            tf.Variable(np.zeros(IMG_INPUT_SHAPE, dtype=DTYPE))
            for _ in range(self.n_frames)
        ]

        self.list_tf_rnn_data = [
            tf.Variable(np.zeros(RNN_INPUT_SHAPE, dtype=DTYPE))
            for _ in range(self.n_frames)
        ]
        self.list_tf_benign_outputs = [
            tf.Variable(np.zeros(shape=MODEL_OUTPUT_SHAPE, dtype=DTYPE))
            for _ in range(self.n_frames)
        ]

        self.tf_desire_input = tf.constant(
            np.zeros(shape=MODEL_DESIRE_INPUT_SHAPE, dtype=DTYPE)
        )
        self.tf_rnn_input = tf.constant(np.zeros(shape=RNN_INPUT_SHAPE, dtype=DTYPE))
        ###
        self.tf_base_color = tf.Variable(np.zeros(3, dtype=DTYPE))

        _expand_base_color = tf.concat(
            [
                [self.tf_base_color[0], self.tf_base_color[0], self.tf_base_color[0]],
                self.tf_base_color,
            ],
            axis=0,
        )
        _expand_base_color = tf.clip_by_value(_expand_base_color, YUV_MIN, YUV_MAX)
        _width_expand_base_color = tf.stack(
            [_expand_base_color for _ in range(IMG_INPUT_SHAPE[2])], axis=-1
        )
        self.ops_tiled_base_color = tf.expand_dims(
            tf.stack(
                [_width_expand_base_color for _ in range(IMG_INPUT_SHAPE[3])], axis=-1
            ),
            axis=0,
        )
        self.tf_poly_inv = tf.constant(compute_path_pinv().astype(DTYPE))  # (4, 50)

        # ops
        self.list_ops_model_img = [
            tf.where(
                tf.is_nan(self.list_tf_model_patches[i]),
                self.list_tf_model_imgs[i],
                tf.clip_by_value(
                    self.list_tf_model_patches[i] + self.ops_tiled_base_color,
                    YUV_MIN,
                    YUV_MAX,
                ),
            )
            for i in range(self.n_frames)
        ]

        self.list_ops_predicts = [
            self.model(
                [
                    self.list_ops_model_img[i],
                    self.tf_desire_input,
                    self.list_tf_rnn_data[i],
                ]
            )
            for i in range(self.n_frames)
        ]

        self.list_ops_patch_update = [
            self.list_tf_model_patches[i].assign(self.buf_grad)
            for i in range(self.n_frames)
        ]
        self.list_ops_img_update = [
            self.list_tf_model_imgs[i].assign(self.buf_img)
            for i in range(self.n_frames)
        ]

        self.list_ops_rnn_update = [
            self.list_tf_rnn_data[i].assign(self.buf_rnn_data)
            for i in range(self.n_frames)
        ]
        self.list_ops_output_update = [
            self.list_tf_benign_outputs[i].assign(self.buf_model_output)
            for i in range(self.n_frames)
        ]
        self.ops_base_color_update = self.tf_base_color.assign(self.yuv_color)

    def _init_tf_variables(self):
        self.sess.run(
            tf.variables_initializer(
                var_list=(
                    self.list_tf_model_imgs
                    + self.list_tf_model_patches
                    + self.list_tf_rnn_data
                    + self.list_tf_benign_outputs
                    + [self.tf_base_color]
                )
            )
        )

    def run(
        self,
        lateral_shift=4,
        starting_meters=60,
        starting_steering_angle=True,
        starting_patch_dir=None,
        starting_patch_epoch=None,
    ):
        logger.debug("enter")

        # initialize car model
        self.car_motion.setup_masks(
            lateral_shift=lateral_shift, starting_meters=starting_meters
        )

        # initialize tf variables
        self._init_tf_variables()

        model_attack_outputs = None

        self.sess.run(
            self.ops_base_color_update,
            feed_dict={self.yuv_color: self.global_base_color},
        )

        # optimization iteration
        # for epoch in tqdm(range(self.n_epoch)):
        epoch = 0
        from skimage.draw import line
        from itertools import product
        from tqdm import tqdm
        best_steer = 100000 if self.is_attack_to_rigth else - 100000
        best_sol = None
        for start, end in tqdm(list(product(
            range(0, self.masked_global_bev_purtabation.shape[1], 10),
            range(0, self.masked_global_bev_purtabation.shape[1], 10)
        ))):

            patch_yuv = np.zeros(self.masked_global_bev_purtabation.shape) * np.nan

            for i in range(4):
                rr, cc = line(0,
                              start + i,
                              self.masked_global_bev_purtabation.shape[0] - 1,
                              end + i)
                patch_yuv[rr, cc] = [1, 1, 1, 1, 0, 0]

            list_patches = self.car_motion.conv_patch2model(
                patch_yuv, self.global_base_color
            )

            [
                self.sess.run(
                    self.list_ops_patch_update[i],
                    {
                        self.buf_grad: np.expand_dims(
                            list_patches[i].transpose((2, 0, 1)), axis=0  # transpose (H, W, ch) ->  (ch, H, W)
                        )
                    },
                )
                for i in range(self.n_frames)
            ]

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
                model_rnn_inputs.append(rnn_input)
                rnn_input = model_output[:, -512:]
            model_outputs = np.vstack(model_outputs)

            logger.debug("update patch imgs")
            [
                self.sess.run(
                    self.list_ops_img_update[i],
                    {self.buf_img: model_img_inputs[i].reshape(IMG_INPUT_SHAPE)},
                )
                for i in range(self.n_frames)
            ]

            logger.debug("update  rnn data")
            if model_attack_outputs is None:
                [
                    self.sess.run(
                        self.list_ops_rnn_update[i],
                        {self.buf_rnn_data: model_rnn_inputs[i].reshape(RNN_INPUT_SHAPE)},
                    )
                    for i in range(self.n_frames)
                ]
            else:
                [
                    self.sess.run(
                        self.list_ops_rnn_update[i],
                        {self.buf_rnn_data: model_attack_outputs[i, -512:].reshape(RNN_INPUT_SHAPE)},
                    )
                    for i in range(self.n_frames)
                ]

            logger.debug("update model output")
            [
                self.sess.run(
                    self.list_ops_output_update[i],
                    {
                        self.buf_model_output: model_outputs[i].reshape(
                            MODEL_OUTPUT_SHAPE
                        )
                    },
                )
                for i in range(self.n_frames)
            ]

            logger.debug("update car trajectory")
            model_attack_outputs = np.vstack(self.sess.run(self.list_ops_predicts))

            self.car_motion.update_trajectory(
                model_attack_outputs, start_steering_angle=starting_steering_angle, add_noise=True
            )
            avg_steer = np.mean(self.car_motion.list_desired_steering_angle)
            if self.is_attack_to_rigth:
                if avg_steer < best_steer:
                    best_steer = avg_steer
                    best_sol = (start, end)
            else:
                if avg_steer > best_steer:
                    best_steer = avg_steer
                    best_sol = (start, end)
            logger.info(f'AAAAAA {self.is_attack_to_rigth} {avg_steer} ({start}, {end}) {best_steer} {best_sol}')

        patch_yuv = np.zeros(self.masked_global_bev_purtabation.shape) * np.nan
        (start, end) = best_sol
        for i in range(5):
            rr, cc = line(0,
                          start + i,
                          self.masked_global_bev_purtabation.shape[0] - 1,
                          end + i)
            patch_yuv[rr, cc] = [1, 1, 1, 1, 0, 0]

        np.save(
            self.result_dir + f"_global_patch_{epoch}",
            patch_yuv,
        )

        np.save(
            self.result_dir + f"_global_masked_patch_{epoch}",
            patch_yuv,
        )
        np.save(
            self.result_dir + f"_global_base_color_{epoch}",
            self.global_base_color,
        )

        model_imgs = np.vstack(self.sess.run(self.list_ops_model_img))
        np.save(self.result_dir + f"model_img_inputs_{epoch}", model_imgs)

        self.last_epoch = epoch
        logger.debug("exit")

    def replay(
        self,
        epoch,
        lateral_shift=4,
        starting_meters=60,
        starting_steering_angle=None,
    ):
        logger.debug("enter")
        output_dir = self.result_dir + 'replay'

        # initialize car model
        self.car_motion.setup_masks(
            lateral_shift=lateral_shift, starting_meters=starting_meters
        )
        # self.list_ops_model_img = self.list_tf_model_imgs

        # initialize tf variables
        self._init_tf_variables()

        self.global_bev_purtabation = np.load(
            self.result_dir + f"_global_patch_{epoch}.npy"
        )
        self.masked_global_bev_purtabation = np.load(
            self.result_dir + f"_global_masked_patch_{epoch}.npy"
        )
        self.global_base_color = np.load(
            self.result_dir + f"_global_base_color_{epoch}.npy"
        )
        model_attack_outputs = None

        logger.debug("start {}".format(epoch))
        logger.debug("calc model ouput")
        list_patches = self.car_motion.conv_patch2model(
            self.masked_global_bev_purtabation, self.global_base_color
        )
        # transpose (H, W, ch) ->  (ch, H, W)
        [
            self.sess.run(
                self.list_ops_patch_update[i],
                {
                    self.buf_grad: np.expand_dims(
                        list_patches[i].transpose((2, 0, 1)), axis=0
                    )
                },
            )
            for i in range(self.n_frames)
        ]
        self.sess.run(
            self.ops_base_color_update,
            feed_dict={self.yuv_color: self.global_base_color},
        )

        model_img_inputs = self.car_motion.calc_model_inputs()

        model_rnn_inputs = []
        model_outputs = []

        def pred_generator():
            rnn_input = np.zeros(RNN_INPUT_SHAPE)
            desire_input = np.zeros(MODEL_DESIRE_INPUT_SHAPE)
            for i in range(self.n_frames):
                model_img_nopatch = self.car_motion.calc_model_inputs_each(i).reshape(IMG_INPUT_SHAPE)
                self.sess.run(
                    self.list_ops_img_update[i],
                    {self.buf_img: model_img_nopatch},
                )
                model_input = self.sess.run(self.list_ops_model_img[i])
                model_output = self.model.predict(
                    [
                        model_input,
                        desire_input,
                        rnn_input,
                    ]
                )
                yield model_output[0]
                model_outputs.append(model_output)
                model_rnn_inputs.append(rnn_input)
                rnn_input = model_output[:, -512:]

        self.car_motion.update_trajectory_gen(
            pred_generator(), start_steering_angle=starting_steering_angle
        )

        np.save(output_dir + f"_global_patch_{epoch}", self.global_bev_purtabation)
        np.save(
            output_dir + f"_global_masked_patch_{epoch}",
            self.masked_global_bev_purtabation,
        )
        np.save(output_dir + f"_global_base_color_{epoch}", self.global_base_color)
        np.save(output_dir + f"model_benign_img_inputs_{epoch}", model_img_inputs)

        model_imgs = np.vstack(self.sess.run(self.list_ops_model_img))
        np.save(output_dir + f"model_outputs_{epoch}", model_attack_outputs)
        np.save(output_dir + f"model_img_inputs_{epoch}", model_imgs)
        np.save(output_dir + f"model_rnn_inputs_{epoch}", model_rnn_inputs)

        objval = -1  # np.mean(self.sess.run([self.ops_obj]))
        if np.isnan(objval):
            raise Exception("obj is nan")

        logger.info(f"epoch: {epoch + 1}, obj: {objval}")
        # with open(output_dir + f"car_motion_{epoch}.pkl", "wb") as f:
        #    pickle.dump(self.car_motion, f, -1)
        with open(output_dir + f"list_patches_{epoch}.pkl", "wb") as f:
            pickle.dump(list_patches, f, -1)
        logger.debug("exit")

    def _agg_gradients(self, list_var_grad):

        model_mask_areas = np.array(
            [m.sum() for m in self.car_motion.get_all_model_masks()]
        )
        weights = model_mask_areas / model_mask_areas.sum()

        list_patch_grad = self.car_motion.conv_model2patch(
            list_var_grad
        )  # zero is missing value
        for i in range(len(list_patch_grad)):
            list_patch_grad[i] *= weights[i]

        tmp = np.stack(list_patch_grad)

        tmp = np.nanmean(tmp, axis=0)
        tmp[np.isnan(tmp)] = 0
        return tmp
