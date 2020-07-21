import numpy as np
import tensorflow as tf

from car_motion_attack.model_output_postprocess import mat_vander
from car_motion_attack.config import DTYPE

N_USE_POINTS = 25
SHIFT_AMOUNT = -10

def compute_path_pinv(l=50):
    deg = 3
    x = np.arange(l * 1.0)
    X = np.vstack(tuple(x ** n for n in range(deg, -1, -1))).T
    pinv = np.linalg.pinv(X)
    return pinv


def poly_fit(pts, stds, v_mat):

    lhs = v_mat / tf.transpose(stds, (1, 0))
    rhs = tf.transpose((pts - pts[0, 0]) / stds, (1, 0))

    poly = tf.linalg.lstsq(lhs, rhs)[:, 0]

    poly = tf.concat([poly, pts[0, :1]], axis=0)
    return poly


def calc_center_path(pred, v_mat):

    MODEL_PATH_DISTANCE = 192
    num_pts = MODEL_PATH_DISTANCE

    path_start = 0
    left_start = MODEL_PATH_DISTANCE * 2
    right_start = MODEL_PATH_DISTANCE * 2 + MODEL_PATH_DISTANCE * 2 + 1
    #lead_start = MODEL_PATH_DISTANCE * 2 + (MODEL_PATH_DISTANCE * 2 + 1) * 2

    pred_path = tf.slice(pred, [0, path_start], [1, num_pts])
    pred_left_lane = tf.slice(pred, [0, left_start], [1, num_pts]) + 1.8
    pred_right_lane = tf.slice(pred, [0, right_start], [1, num_pts]) - 1.8

    path_stds = tf.math.softplus(
        tf.slice(pred, [0, path_start + num_pts], [1, num_pts])
    )
    left_stds = tf.math.softplus(
        tf.slice(pred, [0, left_start + num_pts], [1, num_pts])
    )
    right_stds = tf.math.softplus(
        tf.slice(pred, [0, right_start + num_pts], [1, num_pts])
    )

    left_prob = tf.math.sigmoid(pred[0, left_start + num_pts * 2])
    right_prob = tf.math.sigmoid(pred[0, right_start + num_pts * 2])
    lr_prob = left_prob + right_prob - left_prob * right_prob

    path_poly = poly_fit(pred_path, path_stds, v_mat)
    left_poly = poly_fit(pred_left_lane, left_stds, v_mat)
    right_poly = poly_fit(pred_right_lane, right_stds, v_mat)

    d_poly_lane = (left_prob * left_poly + right_prob * right_poly) / (
        left_prob + right_prob + 0.0001
    )

    d_poly = lr_prob * d_poly_lane + (1.0 - lr_prob) * path_poly
    return d_poly


def loss_func(poly_inv, orig, pred, is_attack_to_rigth=True):

    v_mat = tf.constant(mat_vander.astype(DTYPE))
    poly_center_pred = calc_center_path(pred, v_mat)
    """
    if SHIFT_AMOUNT > 0:
        poly =  - poly_center_pred
    else:
        poly = poly_center_pred
    obj = (
        poly[0]  * (N_USE_POINTS ** 3)
        + poly[1] * (N_USE_POINTS ** 2)
        + poly[2] * (N_USE_POINTS)
    )
    """
    obj = 0
    for i in range(N_USE_POINTS):
        pred_angle = (
            3 * poly_center_pred[0] * (i ** 2)
            + 2 * poly_center_pred[1] * (i)
            + poly_center_pred[2]
        )
        obj += pred_angle  # (target_angle - pred_angle) ** 2 / float(N_USE_POINTS)
    if is_attack_to_rigth:
        return obj # Attack to right
    else:
        return - obj

def NPS_loss(img, use_softmin=False):
    # yuv_palette = np.load(path)
    yuv_palette = np.array([[-0.836, 0, 0]])

    yuv_palette = np.stack([(y, y, y, y, u, v) for y, u, v in yuv_palette])
    tmp = np.tile(yuv_palette, img.shape[2] * img.shape[3]).reshape(
        (yuv_palette.shape[0], img.shape[2], img.shape[3], 6)
    )
    tmp = tmp.transpose([0, 3, 1, 2])[None, :, :, :, :]
    tmp = tf.abs(img - tmp)
    tmp = tf.reduce_sum(tmp, axis=2)

    if use_softmin:
        tmp = -tf.reduce_logsumexp(-tmp * 8, axis=1)
    else:
        tmp = tf.reduce_sum(tf.log1p(tmp), axis=1)

    tmp = tf.reduce_mean(tmp)
    return tmp

def TVLoss(img):
    return tf.image.total_variation(tf.transpose(img[0], (1, 2, 0)))
