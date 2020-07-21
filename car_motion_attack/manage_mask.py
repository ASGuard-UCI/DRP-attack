from logging import getLogger

import cv2
import numpy as np
import pyopencl as cl
from car_motion_attack.utils import prg, queue, ctx
from car_motion_attack.utils import warp_corners, yuv2rgb, rgb2yuv
from car_motion_attack.config import (
    MODEL_IMG_HEIGHT,
    MODEL_IMG_WIDTH,
    IMG_CROP_HEIGHT,
    MODEL_IMG_CH,
    IMG_CROP_WIDTH,
    PIXELS_PER_METER,
)

from car_motion_attack.config import (
    SKY_HEIGHT,
    CAMERA_IMG_WIDTH,
    CAMERA_IMG_HEIGHT,
    #ROI_MAT,
    LATERAL_SHIFT_OFFSET,
    #ROI_MAT_INV,
)

logger = getLogger(__name__)
mf = cl.mem_flags

LANE_INTERVAL = 93
LANE_WIDTH = 1.5
LANE_DOT_LENGTH = 23

class FrameMask:
    def __init__(
        self,
        global_mask,
        bev_mask,
        bev_corners,
        mtx_bev2camera,
        visible_patch_length,
        scale,
        roi_mat,
        roi_mat_inv,
        left_lane_pos=4,
        right_lane_pos=36
    ):

        assert len(bev_mask.shape) == 2
        self.global_mask = global_mask
        self.bev_mask = bev_mask
        self.bev_corners = bev_corners
        self.visible_patch_length = visible_patch_length
        self.scale = scale
        self.roi_mat = roi_mat
        self.roi_mat_inv = roi_mat_inv

        self.left_lane_pos = left_lane_pos
        self.right_lane_pos = right_lane_pos

        self.mtx_bev2camera = mtx_bev2camera
        self.patch_size = global_mask.shape

        self.img_center_width = (
            MODEL_IMG_WIDTH * self.roi_mat[0, 0]
            + MODEL_IMG_HEIGHT * self.roi_mat[0, 1]
            + self.roi_mat[0, 2]
        )
        self.img_center_height = (
            MODEL_IMG_WIDTH * self.roi_mat[1, 0]
            + MODEL_IMG_HEIGHT * self.roi_mat[1, 1]
            + self.roi_mat[1, 2]
        )

        self.lateral_shift = 0
        self.yaw_diff = 0
        self.longitudinal_shift = 0

        self.shifted_rotated_bev_mask = bev_mask
        self.shifted_rotated_bev_corners = bev_corners

        self.camera_mask = self._conv_mask_bev2camera()
        self.model_mask = self._conv_mask_camera2model()
        self.model_ch_mask = np.expand_dims(
            np.stack([self.model_mask] * 6).astype(np.bool), axis=0
        )

        if self.bev_corners is not None:
            self.patch_corners = np.array(
                [
                    [0, 0],
                    [self.patch_size[1] * 2 - 1, 0],
                    [self.patch_size[1] * 2 - 1, self.visible_patch_length * 2 - 1],
                    [0, self.visible_patch_length * 2 - 1],
                ],
                dtype=np.float32,
            )

            self.bev_corners = self.bev_corners.astype(np.float32)
            self.camera_corners = warp_corners(self.mtx_bev2camera, self.bev_corners)
            self.model_corners = self.get_model_corners(self.camera_corners)

            self.mat_model2patch = cv2.getPerspectiveTransform(
                self.model_corners, self.patch_corners
            )
            self.mat_patch2model = cv2.getPerspectiveTransform(
                self.patch_corners, self.model_corners
            )

        else:
            self.bev_corners = None
            self.camera_corners = None
            self.model_corners = None

            self.mat_model2patch = None
            self.mat_patch2model = None

    def _conv_mask_bev2camera(self):
        logger.debug("enter")
        ret = cv2.warpPerspective(
            self.shifted_rotated_bev_mask,
            self.mtx_bev2camera,
            (CAMERA_IMG_WIDTH, CAMERA_IMG_HEIGHT),
        )
        logger.debug("exit")
        return ret  # .astype(np.bool)

    def conv_model2patch(self, mat_grad_yuv):
        if self.mat_model2patch is None:
            return (
                np.ones(
                    shape=(
                        self.global_mask.shape[0],
                        self.global_mask.shape[1],
                        MODEL_IMG_CH,
                    ),
                    dtype=mat_grad_yuv.dtype,
                )
                * np.nan
            )

        mat_grad_rgb = yuv2rgb(mat_grad_yuv)
        mat_grad_rgb = cv2.GaussianBlur(mat_grad_rgb, (5, 5), 0)

        patch_rgb = cv2.warpPerspective(
            mat_grad_rgb,
            self.mat_model2patch,
            (self.patch_size[1] * 2, self.patch_size[0] * 2),
            borderValue=np.nan,
        )
        # patch_rgb[np.isnan(patch_rgb)] = 0
        patch_rgb = cv2.GaussianBlur(patch_rgb, (5, 5), 0)
        # patch_rgb = cv2.blur(patch_rgb, (5, 5))
        patch_yuv = rgb2yuv(patch_rgb)
        #

        """
        patch_yuv = cv2.warpPerspective(mat_grad_yuv,
                                        self.mat_model2patch,
                                        (self.patch_size[1], self.patch_size[0]),
                                        flags=cv2.INTER_NEAREST,
                                        borderValue=np.nan)
        """
        patch_yuv[~self.global_mask] = np.nan

        return patch_yuv

    def conv_patch2model(self, patch, base_color):
        if self.mat_model2patch is None:
            return (
                np.ones(
                    shape=(MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH, MODEL_IMG_CH),
                    dtype=patch.dtype,
                )
                * np.nan
            )

        color_6ch = np.array([base_color[0]] * 4 + [base_color[1]] + [base_color[2]])

        patch_yuv = patch + color_6ch


        for i in range(0, patch_yuv.shape[0], LANE_INTERVAL * self.scale):
            if  self.left_lane_pos is not None:
                patch_yuv[i:i + LANE_DOT_LENGTH * self.scale, self.left_lane_pos * self.scale:int((self.left_lane_pos + LANE_WIDTH) * self.scale), :4] = 0.
            if self.right_lane_pos is not None:
                patch_yuv[i:i + LANE_DOT_LENGTH * self.scale, self.right_lane_pos * self.scale:int((self.right_lane_pos + LANE_WIDTH) * self.scale), :4] = 0.
        visible_patch_yuv = patch_yuv[:self.visible_patch_length]

        # visible_patch_yuv[:, :, :4] += np.random.normal(0, 0.01, size=visible_patch_yuv[:, :, :4].shape)

        visible_patch_rgb = yuv2rgb(visible_patch_yuv).clip(0, 255)
        #visible_patch_rgb[:, :, 0] *= 1.0648  # RED reflection
        #visible_patch_rgb[:, :, 2] *= 0.8967  # BLUE refletion
        # visible_patch_rgb = cv2.GaussianBlur(visible_patch_rgb, (5, 5), 0)
        # visible_patch_rgb = cv2.blur(visible_patch_rgb, (5, 5))

        model_rgb = cv2.warpPerspective(
            visible_patch_rgb,
            self.mat_patch2model,
            (MODEL_IMG_WIDTH * 2, MODEL_IMG_HEIGHT * 2),
            #borderValue=np.nan
            # flags=cv2.INTER_NEAREST
        )
        # mean_model_rgb = model_rgb.mean(axis=2)
        # model_rgb = np.stack([mean_model_rgb] * 3, axis=-1)
        model_rgb = cv2.GaussianBlur(model_rgb, (5, 5), 0)
        # model_rgb = cv2.blur(model_rgb, (5, 5))

        model_yuv = rgb2yuv(model_rgb)
        model_yuv = self._cl_preprocess(model_yuv)
        # model_yuv[:, :, :4] += np.random.normal(0, 0.01, size=model_yuv[:, :, :4].shape)

        model_yuv -= color_6ch
        # model_yuv[:, :, 5:] *= 0 # keep nan is nan
        """
        model_yuv = cv2.warpPerspective(visible_patch_yuv,
                                   self.mat_patch2model,
                                   (MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT))
        """

        model_yuv[~self.model_mask] = np.nan
        return model_yuv

    def _cl_preprocess(self, _yuv_img):

        yuv_img = (_yuv_img * 128 + 128).clip(0, 255).astype(np.uint8)
        yuv_img = np.where(np.isnan(yuv_img), 0, yuv_img)

        y = np.zeros((yuv_img.shape[0] * 2, yuv_img.shape[1] * 2), dtype=np.uint8)

        y[::2, ::2] = yuv_img[:, :, 0]
        y[::2, 1::2] = yuv_img[:, :, 1]
        y[1::2, ::2] = yuv_img[:, :, 2]
        y[1::2, 1::2] = yuv_img[:, :, 3]
        u = yuv_img[:, :, 4].copy()
        v = yuv_img[:, :, 5].copy()

        dist_yuv = np.zeros(
            (MODEL_IMG_CH, MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH), dtype=np.float32
        )
        dist_g_yuv = cl.Buffer(ctx, mf.WRITE_ONLY, dist_yuv.nbytes)

        prg.loadys(
            queue,
            (MODEL_IMG_HEIGHT * MODEL_IMG_WIDTH // 2,),
            None,
            cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y),
            dist_g_yuv,
        )

        prg.loaduv(
            queue,
            (MODEL_IMG_HEIGHT * MODEL_IMG_WIDTH // 8,),
            None,
            cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=u),
            dist_g_yuv,
            np.int32(MODEL_IMG_HEIGHT * MODEL_IMG_WIDTH * 4),
        )
        prg.loaduv(
            queue,
            (MODEL_IMG_HEIGHT * MODEL_IMG_WIDTH // 8,),
            None,
            cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v),
            dist_g_yuv,
            np.int32(MODEL_IMG_HEIGHT * MODEL_IMG_WIDTH * 5),
        )

        cl.enqueue_copy(queue, dist_yuv, dist_g_yuv).wait()

        dist_yuv = np.transpose(dist_yuv, (1, 2, 0))
        dist_yuv = np.where(np.isnan(_yuv_img), np.nan, dist_yuv)

        return dist_yuv

    def reset_mask_area(self, img):
        img[self.model_ch_mask] = 0
        return img

    def get_model_corners(self, _img_corners):
        img_corners = _img_corners.copy()
        img_corners[:, 1] += SKY_HEIGHT
        img_corners = warp_corners(self.roi_mat_inv, img_corners)

        return img_corners.astype(np.float32)

    def _conv_mask_camera2model(self):
        camera_mask = np.vstack([np.zeros((SKY_HEIGHT, CAMERA_IMG_WIDTH), dtype=np.bool),
                                self.camera_mask])

        model_mask = cv2.warpPerspective(camera_mask, self.roi_mat_inv,
                                         (MODEL_IMG_WIDTH * 2, MODEL_IMG_HEIGHT * 2))
        model_mask = cv2.resize(model_mask, (MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT))

        return model_mask > 0.9

    def update_mask(self, lateral_shift, yaw_diff, longitudinal_shift=0):
        self.lateral_shift = lateral_shift
        self.yaw_diff = yaw_diff
        self.longitudinal_shift = longitudinal_shift

        if self.bev_corners is None:
            return

        lat_shift = self.lateral_shift * PIXELS_PER_METER * self.scale
        mtx_lateral_shift = np.float32([[1, 0, lat_shift], [0, 1, 0]])

        shifted_bev_mask = cv2.warpAffine(
            self.bev_mask,
            mtx_lateral_shift,
            (self.bev_mask.shape[1], self.bev_mask.shape[0]),
        )

        lon_shift = self.longitudinal_shift * PIXELS_PER_METER * self.scale
        mtx_lon_shift = np.float32([[1, 0, 0], [0, 1, lon_shift]])

        shifted_bev_mask = cv2.warpAffine(
            shifted_bev_mask,
            mtx_lon_shift,
            (self.bev_mask.shape[1], self.bev_mask.shape[0]),
        )

        cam_origin_shifted = (
            self.bev_mask.shape[1] // 2 - lat_shift + LATERAL_SHIFT_OFFSET,
            self.bev_mask.shape[0] - 1 - lon_shift,

        )

        mtx_rotation = cv2.getRotationMatrix2D(cam_origin_shifted, -yaw_diff, 1)

        self.shifted_rotated_bev_mask = cv2.warpAffine(
            shifted_bev_mask,
            mtx_rotation,
            (self.bev_mask.shape[1], self.bev_mask.shape[0]),
        )

        shifted_corners = warp_corners(mtx_lateral_shift, self.bev_corners)
        shifted_corners = warp_corners(mtx_lon_shift, shifted_corners)
        self.shifted_rotated_bev_corners = warp_corners(mtx_rotation, shifted_corners)

        self.camera_mask = self._conv_mask_bev2camera()
        self.model_mask = self._conv_mask_camera2model()
        self.model_ch_mask = np.expand_dims(
            np.stack([self.model_mask] * 6).astype(np.bool), axis=0
        )

        self.camera_corners = warp_corners(
            self.mtx_bev2camera, self.shifted_rotated_bev_corners
        )
        self.model_corners = self.get_model_corners(self.camera_corners)
        self.mat_model2patch = cv2.getPerspectiveTransform(
            self.model_corners, self.patch_corners
        )
        self.mat_patch2model = cv2.getPerspectiveTransform(
            self.patch_corners, self.model_corners
        )
