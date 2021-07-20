import os
import sys
from PIL import Image
import io
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
from car_motion_attack.utils import AdamOpt, yuv2rgb, rgb2yuv
from car_motion_attack.models import create_model
from car_motion_attack.replay_bicycle import ReplayBicycle
from car_motion_attack.loss import compute_path_pinv, loss_func
from car_motion_attack.config import (DTYPE, PIXELS_PER_METER, SKY_HEIGHT, IMG_INPUT_SHAPE,
                                      IMG_INPUT_MASK_SHAPE, RNN_INPUT_SHAPE,
                                      MODEL_DESIRE_INPUT_SHAPE, MODEL_OUTPUT_SHAPE,
                                      YUV_MIN, YUV_MAX
                                      )
from scipy import ndimage
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
        learning_rate_patch=1.0e-1,
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

        from car_motion_attack.magnet import DenoisingAutoEncoder, MAP_MAGNET, IMG_CROP_HEIGHT, IMG_CROP_WIDTH
        tmp = DenoisingAutoEncoder((IMG_CROP_HEIGHT, IMG_CROP_WIDTH, 3),                                                                                                                                           
                            MAP_MAGNET[sys.argv[2]]['model'],     
                            model_dir=MAP_MAGNET[sys.argv[2]]['path'],                                                                                                                            
                            v_noise=0.1,                                                                                                                                                                    
                            activation='relu',                                                                                                                                                              
                            reg_strength=1e-9)

        tmp.load('best_weights.hdf5')
        self.magnet_model = tmp.model

        
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
        """
        self.ops_obj_shifting = sum(
            loss_func(
                self.tf_poly_inv,
                self.list_tf_benign_outputs[i],
                self.list_ops_predicts[i],
                is_attack_to_rigth=self.is_attack_to_rigth
            )
            for i in tqdm(range(self.n_frames), desc="loss")
        )

        # self.ops_obj_l1 = sum(
        #    [
        #        tf.abs(
        #            tf.where(
        #                tf.is_nan(self.list_tf_model_patches[i]),
        #                tf.zeros_like(self.list_tf_model_patches[i]),
        #                self.list_tf_model_patches[i],
        #            )
        #        )
        #        for i in range(self.n_frames)
        #    ]
        # )

        self.ops_obj_l2 = sum(
            tf.nn.l2_loss(
                tf.where(
                    tf.is_nan(self.list_tf_model_patches[i]),
                    tf.zeros_like(self.list_tf_model_patches[i]),
                    self.list_tf_model_patches[i],
                )
            )
            for i in range(self.n_frames)
        )

        # self.ops_obj_tvloss = sum(
        #    tf.nn.l2_loss(
        #        tf.where(
        #            tf.is_nan(self.list_tf_model_patches[i]),
        #            tf.zeros_like(self.list_tf_model_patches[i]),
        #            self.list_tf_model_patches[i],
        #        )
        #    )
        #    for i in range(self.n_frames)
        # )

        # + self.ops_obj_l1 * 0.01# + 0.01 * self.ops_obj_tvloss
        self.ops_obj = self.ops_obj_shifting + (self.l2_weight * self.ops_obj_l2)

        self.list_ops_gradients = tf.gradients(
            self.ops_obj, self.list_tf_model_patches + [self.tf_base_color]
        )
        """
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

        adam_patch = AdamOpt(
            self.global_bev_purtabation.shape, lr=self.learning_rate_patch
        )
        # adam_color = AdamOpt(self.global_base_color.shape, lr=self.learning_rate_color)
        model_attack_outputs = None
        if starting_patch_dir is not None:
            self.global_bev_purtabation = np.load(
                starting_patch_dir + f"_global_patch_{starting_patch_epoch}.npy"
            )
            self.masked_global_bev_purtabation = np.load(
                starting_patch_dir + f"_global_masked_patch_{starting_patch_epoch}.npy"
            )
            self.global_base_color = np.load(
                starting_patch_dir + f"_global_base_color_{starting_patch_epoch}.npy"
            )
        self.sess.run(
            self.ops_base_color_update,
            feed_dict={self.yuv_color: self.global_base_color},
        )
        # optimization iteration
        for epoch in tqdm(range(self.n_epoch)):

            logger.debug("start {}".format(epoch))

            if epoch % 100 == 0:
                adam_patch.lr *= 0.9
                #adam_color.lr *= 0.9

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

            logger.debug("calc gradients")
            _grads = self.sess.run(self.list_ops_gradients)

            list_var_grad = [g[0].transpose((1, 2, 0)) for g in _grads[:-1]]
            base_color_grad = _grads[-1]
            #base_color_grad[1:] = 0

            logger.debug("conv gradients -> patch")
            logger.debug("agg patch grads")
            patch_grad = self._agg_gradients(list_var_grad)

            logger.debug("update global purtabation")

            logger.debug(
                f"global_base_color: {self.global_base_color} {base_color_grad}"
            )

            self.global_bev_purtabation -= adam_patch.update(patch_grad)
            self.global_bev_purtabation = self.global_bev_purtabation.clip(
                0, - self.base_color
            )

            self.global_bev_purtabation[:, :, 4:] = 0

            patch_diff = self.global_bev_purtabation.sum(axis=2)
            threshold = np.percentile(patch_diff, 100 - self.perturbable_area_ratio)
            mask_bev_purtabation = patch_diff >= threshold

            self.masked_global_bev_purtabation = self.global_bev_purtabation.copy()
            self.masked_global_bev_purtabation[~mask_bev_purtabation] = 0.

            if np.isnan(self.global_bev_purtabation.sum()):
                raise Exception("patch encouter nan")

            logger.debug("apply global purtabation to each frame")
            list_patches = self.car_motion.conv_patch2model(
                self.masked_global_bev_purtabation, self.global_base_color
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
            self.sess.run(
                self.ops_base_color_update,
                feed_dict={self.yuv_color: self.global_base_color},
            )

            if epoch % 100 == 0 and epoch > 0:
                np.save(
                    self.result_dir + f"_global_patch_{epoch}",
                    self.global_bev_purtabation,
                )

                np.save(
                    self.result_dir + f"_global_masked_patch_{epoch}",
                    self.masked_global_bev_purtabation,
                )
                np.save(
                    self.result_dir + f"_global_base_color_{epoch}",
                    self.global_base_color,
                )
                np.save(
                    self.result_dir + f"model_benign_img_inputs_{epoch}",
                    model_img_inputs,
                )

                model_imgs = np.vstack(self.sess.run(self.list_ops_model_img))
                np.save(
                    self.result_dir + f"model_outputs_{epoch}", model_attack_outputs
                )
                np.save(self.result_dir + f"model_img_inputs_{epoch}", model_imgs)
                np.save(self.result_dir + f"model_rnn_inputs_{epoch}", model_rnn_inputs)
                objval = np.mean(self.sess.run([self.ops_obj]))
                if np.isnan(objval):
                    raise Exception("obj is nan")

                logger.info(
                    f"save epoch: {epoch + 1}, obj: {objval} total_lat: {self.car_motion.list_total_lateral_shift} desired: {self.car_motion.list_desired_steering_angle}"
                )

                if (
                    (self.is_attack_to_rigth and self.car_motion.list_lateral_shift_openpilot[-1] < - self.target_deviation) or
                    ((not self.is_attack_to_rigth)
                     and self.car_motion.list_lateral_shift_openpilot[-1] > self.target_deviation)
                ):
                    logger.info(
                        f"Reached target deviation: {epoch + 1}, obj: {objval} total_lat: {self.car_motion.list_lateral_shift_openpilot[-1]}"
                    )
                    self.last_epoch = epoch
                    break
                # with open(self.result_dir + f"car_motion_{epoch}.pkl", "wb") as f:
                #    pickle.dump(self.car_motion, f, -1)
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

        for _ in range(5):
            logger.debug("start {}".format(epoch))
            logger.debug("calc model ouput")
            list_patches = self.car_motion.conv_patch2model(
                self.masked_global_bev_purtabation, self.global_base_color
            )
            self.sess.run(
                self.ops_base_color_update,
                feed_dict={self.yuv_color: self.global_base_color},
            )

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

            def magnet_defense(img, patch):
                _img = np.where(np.isnan(patch), img.reshape(IMG_INPUT_SHAPE)[0].transpose((1, 2, 0)), patch)
                rgb = yuv2rgb(_img) / 255 #.astype(np.uint8)
                rgb = self.magnet_model.predict(np.expand_dims(rgb, axis=0))[0] * 255
                yuv = rgb2yuv(rgb.astype(np.uint8))
                _patch = np.where(~np.isnan(patch), yuv, np.nan)
                return yuv.transpose((2, 0, 1)).flatten(), _patch

            tmp = [
                magnet_defense(model_img_inputs[i], list_patches[i])
                for i in range(self.n_frames)
            ]
            model_img_inputs = [t[0] for t in tmp]
            list_patches = [t[1] for t in tmp]

            logger.debug("update benign imgs")
            [
                self.sess.run(
                    self.list_ops_img_update[i],
                    {self.buf_img: model_img_inputs[i].reshape(IMG_INPUT_SHAPE)},
                )
                for i in range(self.n_frames)
            ]

            logger.debug("update benign rnn data")
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

            logger.debug("update benign model output")
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
            logger.debug("update patch")

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
            logger.debug("update trajectory")
            model_attack_outputs = np.vstack(self.sess.run(self.list_ops_predicts))

            self.car_motion.update_trajectory(
                model_attack_outputs, start_steering_angle=starting_steering_angle
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
        # if np.isnan(objval):
        #    raise Exception("obj is nan")

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
