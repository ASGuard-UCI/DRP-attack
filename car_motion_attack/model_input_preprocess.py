import os
from logging import getLogger

import cv2
import numpy as np
import pyopencl as cl

from car_motion_attack.utils import rgb2yuv, transform_scale_buffer
from car_motion_attack.utils import prg, queue, ctx
from car_motion_attack.config import (
    MODEL_IMG_HEIGHT,
    MODEL_IMG_WIDTH,
    IMG_CROP_HEIGHT,
    IMG_CROP_WIDTH,
    MODEL_IMG_HEIGHT,
    MODEL_IMG_WIDTH,
    MODEL_IMG_CH,
    #ROI_MAT, ROI_MAT_INV
)

logger = getLogger(__name__)

TEMP_PATH = os.path.dirname(os.path.abspath(__file__))
mf = cl.mem_flags


class ModelInPreprocess:


    def __init__(
        self,
        roi_mat
    ):
        self.mtx_frame2camera_y = roi_mat.astype(np.float32)
        self.mtx_frame2camera_y_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.mtx_frame2camera_y)
        self.mtx_frame2camera_uv = transform_scale_buffer(roi_mat, 0.5).astype(np.float32)
        self.mtx_frame2camera_uv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.mtx_frame2camera_uv)

    def visiond_prep(self, _yuv_img):
        yuv_img = (_yuv_img * 128 + 128).clip(0, 255).astype(np.uint8)
        y = np.zeros((yuv_img.shape[0] * 2, yuv_img.shape[1] * 2), dtype=np.uint8)

        y[::2, ::2] = yuv_img[:, :, 0]
        y[::2, 1::2] = yuv_img[:, :, 1]
        y[1::2, ::2] = yuv_img[:, :, 2]
        y[1::2, 1::2] = yuv_img[:, :, 3]
        u = yuv_img[:, :, 4].copy()
        v = yuv_img[:, :, 5].copy()

        dist_y = np.zeros((MODEL_IMG_HEIGHT * 2, MODEL_IMG_WIDTH * 2), dtype=np.uint8)
        dist_g_y = cl.Buffer(ctx, mf.WRITE_ONLY, dist_y.nbytes)

        dist_u = np.zeros((MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH), dtype=np.uint8)
        dist_g_u = cl.Buffer(ctx, mf.WRITE_ONLY, dist_u.nbytes)

        dist_v = np.zeros((MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH), dtype=np.uint8)
        dist_g_v = cl.Buffer(ctx, mf.WRITE_ONLY, dist_v.nbytes)

        prg.warpPerspective(queue, y.shape, None,
                            cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y),
                            np.int32(y.shape[1]),
                            np.int32(0),
                            np.int32(y.shape[0]),
                            np.int32(y.shape[1]),
                            dist_g_y,
                            np.int32(MODEL_IMG_WIDTH * 2),
                            np.int32(0),
                            np.int32(MODEL_IMG_HEIGHT * 2),
                            np.int32(MODEL_IMG_WIDTH * 2),                    
                            self.mtx_frame2camera_y_buf
                            )

        prg.warpPerspective(queue, u.shape, None,
                            cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=u),
                            np.int32(u.shape[1]),
                            np.int32(0),
                            np.int32(u.shape[0]),
                            np.int32(u.shape[1]),
                            dist_g_u,
                            np.int32(MODEL_IMG_WIDTH),
                            np.int32(0),
                            np.int32(MODEL_IMG_HEIGHT),
                            np.int32(MODEL_IMG_WIDTH),                    
                             self.mtx_frame2camera_uv_buf
                            )
        prg.warpPerspective(queue, y.shape, None,
                            cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v),
                            np.int32(v.shape[1]),
                            np.int32(0),
                            np.int32(v.shape[0]),
                            np.int32(v.shape[1]),
                            dist_g_v,
                            np.int32(MODEL_IMG_WIDTH),
                            np.int32(0),
                            np.int32(MODEL_IMG_HEIGHT),
                            np.int32(MODEL_IMG_WIDTH),                     
                             self.mtx_frame2camera_uv_buf
                            )

        cl.enqueue_copy(queue, dist_y, dist_g_y)
        cl.enqueue_copy(queue, dist_u, dist_g_u)
        cl.enqueue_copy(queue, dist_v, dist_g_v)

        #np.save('aaa', dist_y)

        dist_yuv = np.zeros((MODEL_IMG_CH * MODEL_IMG_HEIGHT * MODEL_IMG_WIDTH), dtype=np.float32)
        dist_g_yuv = cl.Buffer(ctx, mf.WRITE_ONLY, dist_yuv.nbytes)

        prg.loadys(queue, (MODEL_IMG_HEIGHT * MODEL_IMG_WIDTH // 2,), None,
                    cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dist_y),
                    dist_g_yuv)

        prg.loaduv(queue, (MODEL_IMG_HEIGHT * MODEL_IMG_WIDTH // 8,), None,
                    cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dist_u),
                    dist_g_yuv,
                    np.int32(MODEL_IMG_HEIGHT * MODEL_IMG_WIDTH * 4)
                    )
        prg.loaduv(queue, (MODEL_IMG_HEIGHT * MODEL_IMG_WIDTH // 8,), None,
                    cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dist_v),
                    dist_g_yuv,
                    np.int32(MODEL_IMG_HEIGHT * MODEL_IMG_WIDTH * 5)
                    )

        cl.enqueue_copy(queue, dist_yuv, dist_g_yuv).wait()

        return dist_yuv

    def rgb_to_modelin(self, rgb_img):
        #roi_rgb = self._visiond_prep_roi(rgb_img)
        yuv_img = rgb2yuv(rgb_img)
        model_in_dat = self.visiond_prep(yuv_img)
        return model_in_dat

