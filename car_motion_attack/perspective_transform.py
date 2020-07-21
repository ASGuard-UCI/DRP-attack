import cv2
import math
import numpy as np
from logging import getLogger
from car_motion_attack.config import PIXELS_PER_METER, SKY_HEIGHT, BEV_BASE_HEIGHT, BEV_BASE_WIDTH, LATERAL_SHIFT_OFFSET

logger = getLogger(__name__)


class PerspectiveTransform:
    def __init__(self, bgr_all_img, mtx_src2dist, mtx_dist2src, scale, yaw_diff_offset=0):

        self.bgr_all_img = bgr_all_img
        self.img_all = cv2.cvtColor(self.bgr_all_img, cv2.COLOR_BGR2RGB)

        bgr_img = bgr_all_img[SKY_HEIGHT:]
        self.img_raw = bgr_img

        self.img = cv2.cvtColor(self.img_raw, cv2.COLOR_BGR2RGB)
        self.frame_height, self.frame_width, _ = bgr_img.shape
        self.scale = scale

        self.lateral_shift = 0
        self.longitudinal_shift = 0
        self.yaw_diff = 0
        self.yaw_diff_offset = yaw_diff_offset

        self.cam_origin = (BEV_BASE_HEIGHT * scale - 1, (BEV_BASE_WIDTH // 2) * scale)

        self.mtx_src2dist = mtx_src2dist
        self.mtx_dist2src = mtx_dist2src

        self.bev_image = self.conv_bev_image(self.img)
        self.bev_height, self.bev_width, _ = self.bev_image.shape

        self.shifted_roatated_bev_image = self.bev_image
        self.shifted_roatated_camera_image = self.img

    def conv_bev_image(self, img):
        non_zoom_img = cv2.warpPerspective(
            img,
            self.mtx_src2dist,
            (BEV_BASE_WIDTH * self.scale, (BEV_BASE_HEIGHT) * self.scale),
        )
        # zoom_img = non_zoom_img[450 * self.scale, 575 * self.scale: 625 * self.scale, :]
        return non_zoom_img

    def conv_camera_image(self, img):
        return cv2.warpPerspective(
            img, self.mtx_dist2src, (self.frame_width, self.frame_height)
        )

    def get_zoomed_image(self, img):
        return img[450 * self.scale:, 575 * self.scale: 625 * self.scale, :]

    def update_perspective(self, lateral_shift, yaw_diff, longitudinal_shift=0):
        logger.debug("enter")
        self.lateral_shift = lateral_shift
        self.longitudinal_shift = longitudinal_shift
        self.yaw_diff = yaw_diff

        bev_img = self._update_shifted_roatated_bev_image()
        camera_img = self.conv_camera_image(bev_img)

        # debug
        # fig, ax = plt.subplots(figsize=(15, 15))
        # ax.imshow(camera_img)
        # plt.title('shifted roatated camera')
        # plt.show()
        ###
        self.shifted_roatated_camera_image = camera_img
        logger.debug("exit")

    def create_shifted_roatated_bev_image(self, bev_image=None):
        if bev_image is None:
            bev_image = self.bev_image
        lat_shift = self.lateral_shift * PIXELS_PER_METER * self.scale

        mtx_lateral_shift = np.float32([[1, 0, lat_shift],
                                        [0, 1, 0]])
        shifted_bev_img = cv2.warpAffine(
            bev_image, mtx_lateral_shift, (self.bev_width, self.bev_height)
        )

        lon_shift = self.longitudinal_shift * PIXELS_PER_METER * self.scale

        mtx_lon_shift = np.float32([[1, 0, 0],
                                    [0, 1, lon_shift]])
        shifted_bev_img = cv2.warpAffine(
            shifted_bev_img, mtx_lon_shift, (self.bev_width, self.bev_height)
        )

        # debug
        # fig, ax = plt.subplots(figsize=(15, 15))
        # ax.imshow(self.get_zoomed_image(self.bev_image))
        # plt.title('original bev')
        # plt.show()

        # fig, ax = plt.subplots(figsize=(15, 15))
        # ax.imshow(self.get_zoomed_image(shifted_bev_img))
        # plt.title('shifted bev')
        # plt.show()
        ###

        cam_origin_shifted = (self.cam_origin[1] - lat_shift, self.cam_origin[0] - lon_shift)

        mtx_rotation = cv2.getRotationMatrix2D(cam_origin_shifted, - self.yaw_diff, 1)

        shifted_rotated_bev_img = cv2.warpAffine(
            shifted_bev_img, mtx_rotation, (self.bev_width, self.bev_height)
        )
        # debug
        # fig, ax = plt.subplots(figsize=(15, 15))
        # ax.imshow(self.get_zoomed_image(shifted_rotated_bev_img))
        # plt.title('shifted roatated bev')
        # plt.show()
        ###
        #self.shifted_roatated_bev_image = shifted_rotated_bev_img
        return shifted_rotated_bev_img

    def get_sky_img(self, mergin=5):

        img = self.img_all[:SKY_HEIGHT + mergin]
        h, w = img.shape[:2]

        lat_shift = (self.lateral_shift + math.radians(self.yaw_diff) * 100) * PIXELS_PER_METER

        mtx_lateral_shift = np.float32([[1, 0, lat_shift], [0, 1, 0]])
        img = cv2.warpAffine(
            img, mtx_lateral_shift, (w, h)
        )

        sky_img = self.img_all[:SKY_HEIGHT + mergin][:-25]

        #h, w = sky_img.shape[:2]
        #mtx_lateral_shift = np.float32([[1, 0, lat_shift * 1], [0, 1, 0]])
        # sky_img = cv2.warpAffine(
        #    sky_img, mtx_lateral_shift, (w, h)
        # )

        #sky_img[:] = 0
        ground_img = img[-25:]

        h, w = ground_img.shape[:2]

        src = np.array([[0.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0]], np.float32)
        dest = src.copy()
        dest[:, 0] += (- lat_shift / h * (h - src[:, 1])).astype(np.float32)
        affine = cv2.getAffineTransform(src, dest)
        ground_img = cv2.warpAffine(ground_img, affine, (w, h))

        img = np.vstack([sky_img, ground_img])

        return img

    def _update_shifted_roatated_bev_image(self):

        self.shifted_roatated_bev_image = self.create_shifted_roatated_bev_image()
        return self.shifted_roatated_bev_image
