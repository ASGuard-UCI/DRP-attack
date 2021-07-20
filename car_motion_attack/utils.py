import os
import ctypes

from numpy.ctypeslib import ndpointer
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import cv2
from numba import jit, prange


def warp_coord(M, coord):
    if M.shape[0] == 3:
        x = (M[0, 0]*coord[0] + M[0, 1]*coord[1] + M[0, 2])/(M[2, 0]*coord[0] + M[2, 1]*coord[1] + M[2, 2])
        y = (M[1, 0]*coord[0] + M[1, 1]*coord[1] + M[1, 2])/(M[2, 0]*coord[0] + M[2, 1]*coord[1] + M[2, 2])
    else:
        x = M[0, 0]*coord[0] + M[0, 1]*coord[1] + M[0, 2]
        y = M[1, 0]*coord[0] + M[1, 1]*coord[1] + M[1, 2]

    warped_coord = np.array([x, y])
    return warped_coord


def warp_corners(mtx, corners):
    # corners = np.array([[w_lo, h_lo],
    #                       [w_hi, h_lo],
    #                       [w_hi, h_hi],
    #                       [w_lo, h_hi]])
    warped_corners = np.zeros((corners.shape[0], 2))
    for i, c in enumerate(corners):
        warped_corners[i] = warp_coord(mtx, c)
    return warped_corners


def ecef2geodetic(ecef, radians=False):
    """
    Convert ECEF coordinates to geodetic using ferrari's method
    """
    a = 6378137
    b = 6356752.3142
    esq = 6.69437999014 * 0.001
    e1sq = 6.73949674228 * 0.001

    # Save shape and export column
    ecef = np.atleast_1d(ecef)
    input_shape = ecef.shape
    ecef = np.atleast_2d(ecef)
    x, y, z = ecef[:, 0], ecef[:, 1], ecef[:, 2]

    ratio = 1.0 if radians else (180.0 / np.pi)

    # Conver from ECEF to geodetic using Ferrari's methods
    # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#Ferrari.27s_solution
    r = np.sqrt(x * x + y * y)
    Esq = a * a - b * b
    F = 54 * b * b * z * z
    G = r * r + (1 - esq) * z * z - esq * Esq
    C = (esq * esq * F * r * r) / (pow(G, 3))
    S = np.cbrt(1 + C + np.sqrt(C * C + 2 * C))
    P = F / (3 * pow((S + 1 / S + 1), 2) * G * G)
    Q = np.sqrt(1 + 2 * esq * esq * P)
    r_0 = -(P * esq * r) / (1 + Q) + np.sqrt(0.5 * a * a*(1 + 1.0 / Q) -
                                             P * (1 - esq) * z * z / (Q * (1 + Q)) - 0.5 * P * r * r)
    U = np.sqrt(pow((r - esq * r_0), 2) + z * z)
    V = np.sqrt(pow((r - esq * r_0), 2) + (1 - esq) * z * z)
    Z_0 = b * b * z / (a * V)
    h = U * (1 - b * b / (a * V))
    lat = ratio*np.arctan((z + e1sq * Z_0) / r)
    lon = ratio*np.arctan2(y, x)

    # stack the new columns and return to the original shape
    geodetic = np.column_stack((lat, lon, h))
    return geodetic.reshape(input_shape)

def transform_scale_buffer(orig_mat, s):

    transform_out = np.array(
        [[1.0/s, 0.0, 0.5],
        [0.0, 1.0/s, 0.5],
        [0.0, 0.0, 1.0]]
    )

    transform_in = np.array(
        [[s,  0.0, -0.5*s],
        [0.0, s, -0.5*s],
        [0.0, 0.0, 1.0]]
    )

    return np.dot(transform_in, np.dot(orig_mat, transform_out))


MAT_YUV2RGB = np.array([[1.1643, 0, 1.5958],
                        [1.1643, - 0.39173, - 0.81290],
                        [1.1643, 2.017, 0]]).T
MAT_RGB2YUV = np.linalg.inv(MAT_YUV2RGB)


def _yuv2rgb(new_img):
    img = new_img * 128
    img[:, :, :4] += 112
    pix = np.zeros((new_img.shape[0] * 2, new_img.shape[1] * 2, 3))
    pix[::2, ::2] = img[:, :, np.array([0, 4, 5])].dot(MAT_YUV2RGB)
    pix[::2, 1::2] = img[:, :, np.array([1, 4, 5])].dot(MAT_YUV2RGB)
    pix[1::2, ::2] = img[:, :, np.array([2, 4, 5])].dot(MAT_YUV2RGB)
    pix[1::2, 1::2] = img[:, :, np.array([3, 4, 5])].dot(MAT_YUV2RGB)
    return pix


def _rgb2yuv(pix):
    img = np.zeros((pix.shape[0] // 2, pix.shape[1] // 2, 6))
    img[:, :, [0, 4, 5]] = pix[::2, ::2].dot(MAT_RGB2YUV)
    img[:, :, [1, 4, 5]] = pix[::2, 1::2].dot(MAT_RGB2YUV)
    img[:, :, [2, 4, 5]] = pix[1::2, ::2].dot(MAT_RGB2YUV)
    img[:, :, [3, 4, 5]] = pix[1::2, 1::2].dot(MAT_RGB2YUV)

    img[:, :, :4] -= 112
    img = img / 128
    return img


@jit(nopython=True, parallel=True, nogil=True)
def yuv2rgb(new_img):
    pix = np.zeros((new_img.shape[0] * 2, new_img.shape[1] * 2, 3))

    for i in prange(new_img.shape[0]):
        for j in prange(new_img.shape[1]):
            for k in prange(4):
                Y_val = new_img[i, j, k] * 128 + 112
                U_val = new_img[i, j, 4] * 128
                V_val = new_img[i, j, 5] * 128

                B = 1.1643 * (Y_val) + 2.017 * (U_val)
                G = 1.1643 * (Y_val) - 0.81290 * (V_val) - 0.39173 * (U_val)
                R = 1.1643 * (Y_val) + 1.5958 * (V_val)

                p = (R, G, B)
                if k == 0:
                    pix[i * 2, j * 2] = p
                elif k == 1:
                    pix[i * 2, j * 2 + 1] = p
                elif k == 2:
                    pix[i * 2 + 1, j * 2] = p
                elif k == 3:
                    pix[i * 2 + 1, j * 2 + 1] = p
    return pix


@jit(nopython=True, parallel=True, nogil=True)
def rgb2yuv(pix):
    img = np.zeros((pix.shape[0] // 2, pix.shape[1] // 2, 6))

    for i in prange(img.shape[0]):
        for j in prange(img.shape[1]):
            U_val = 0
            V_val = 0
            for k in prange(4):
                if k == 0:
                    p = pix[i * 2, j * 2]
                elif k == 1:
                    p = pix[i * 2, j * 2 + 1]
                elif k == 2:
                    p = pix[i * 2 + 1, j * 2]
                elif k == 3:
                    p = pix[i * 2 + 1, j * 2 + 1]

                Y_val = 0.25681631 * p[0] + 0.50415484 * p[1] + 0.09791402 * p[2]
                U = -0.14824553 * p[0] + -0.29102007 * p[1] + 0.4392656 * p[2]
                V = 0.43927107 * p[0] + -0.36783273 * p[1] - 0.07143833 * p[2]

                img[i, j, k] = (Y_val - 112) / 128
                U_val += U
                V_val += V

            img[i, j, 4] = U_val / 4 / 128
            img[i, j, 5] = V_val / 4 / 128
    return img


class AdamOpt:

    def __init__(self, size, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, dtype=np.float32):
        self.exp_avg = np.zeros(size, dtype=dtype)
        self.exp_avg_sq = np.zeros(size, dtype=dtype)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.step = 0

    def update(self, grad):

        self.step += 1

        bias_correction1 = 1 - self.beta1 ** self.step
        bias_correction2 = 1 - self.beta2 ** self.step

        self.exp_avg = self.beta1 * self.exp_avg + (1 - self.beta1) * grad
        self.exp_avg_sq = self.beta2 * self.exp_avg_sq + (1 - self.beta2) * (grad ** 2)

        denom = (np.sqrt(self.exp_avg_sq) / np.sqrt(bias_correction2)) + self.eps

        step_size = self.lr / bias_correction1

        return step_size / denom * self.exp_avg

import pyopencl as cl
for platform in cl.get_platforms():
    device = platform.get_devices()[0]
    try:
        ctx = cl.Context([device])
        break
    except:
        continue

queue = cl.CommandQueue(ctx)


prg = cl.Program(ctx, """
#define INTER_BITS 5
#define INTER_TAB_SIZE (1 << INTER_BITS)
#define INTER_SCALE 1.f / INTER_TAB_SIZE

#define INTER_REMAP_COEF_BITS 15
#define INTER_REMAP_COEF_SCALE (1 << INTER_REMAP_COEF_BITS)

__kernel void warpPerspective(__global const uchar * src, //0
                              int src_step, //1
                              int src_offset, //2
                              int src_rows,  //3
                              int src_cols, //4
                              __global uchar * dst, //5
                              int dst_step,  //6
                              int dst_offset,  //7
                              int dst_rows,  //8
                              int dst_cols, //9
                              __constant float * M) //10
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        float X0 = M[0] * dx + M[1] * dy + M[2];
        float Y0 = M[3] * dx + M[4] * dy + M[5];
        float W = M[6] * dx + M[7] * dy + M[8];
        W = W != 0.0f ? INTER_TAB_SIZE / W : 0.0f;
        int X = rint(X0 * W), Y = rint(Y0 * W);

        short sx = convert_short_sat(X >> INTER_BITS);
        short sy = convert_short_sat(Y >> INTER_BITS);
        short ay = (short)(Y & (INTER_TAB_SIZE - 1));
        short ax = (short)(X & (INTER_TAB_SIZE - 1));

        int v0 = (sx >= 0 && sx < src_cols && sy >= 0 && sy < src_rows) ?
            convert_int(src[mad24(sy, src_step, src_offset + sx)]) : 0;
        int v1 = (sx+1 >= 0 && sx+1 < src_cols && sy >= 0 && sy < src_rows) ?
            convert_int(src[mad24(sy, src_step, src_offset + (sx+1))]) : 0;
        int v2 = (sx >= 0 && sx < src_cols && sy+1 >= 0 && sy+1 < src_rows) ?
            convert_int(src[mad24(sy+1, src_step, src_offset + sx)]) : 0;
        int v3 = (sx+1 >= 0 && sx+1 < src_cols && sy+1 >= 0 && sy+1 < src_rows) ?
            convert_int(src[mad24(sy+1, src_step, src_offset + (sx+1))]) : 0;

        float taby = 1.f/INTER_TAB_SIZE*ay;
        float tabx = 1.f/INTER_TAB_SIZE*ax;

        int dst_index = mad24(dy, dst_step, dst_offset + dx);

        int itab0 = convert_short_sat_rte( (1.0f-taby)*(1.0f-tabx) * INTER_REMAP_COEF_SCALE );
        int itab1 = convert_short_sat_rte( (1.0f-taby)*tabx * INTER_REMAP_COEF_SCALE );
        int itab2 = convert_short_sat_rte( taby*(1.0f-tabx) * INTER_REMAP_COEF_SCALE );
        int itab3 = convert_short_sat_rte( taby*tabx * INTER_REMAP_COEF_SCALE );

        int val = v0 * itab0 +  v1 * itab1 + v2 * itab2 + v3 * itab3;

        uchar pix = convert_uchar_sat((val + (1 << (INTER_REMAP_COEF_BITS-1))) >> INTER_REMAP_COEF_BITS);
        dst[dst_index] = pix;
    }
}

#define TRANSFORMED_WIDTH 512
#define TRANSFORMED_HEIGHT 256

#define UV_SIZE ((TRANSFORMED_WIDTH/2)*(TRANSFORMED_HEIGHT/2))

__kernel void loadys(__global uchar8 const * const Y,
                     __global float * out)
{
    const int gid = get_global_id(0);
    const int ois = gid * 8;
    const int oy = ois / TRANSFORMED_WIDTH;
    const int ox = ois % TRANSFORMED_WIDTH;

    const uchar8 ys = Y[gid];

    // y = (x - 128) / 128
    const float8 ysf = (convert_float8(ys) - 128.f) * 0.0078125f;

    // 02
    // 13

    __global float* outy0;
    __global float* outy1;
    if ((oy & 1) == 0) {
      outy0 = out; //y0
      outy1 = out + UV_SIZE*2; //y2
    } else {
      outy0 = out + UV_SIZE; //y1
      outy1 = out + UV_SIZE*3; //y3
    }

    vstore4(ysf.s0246, 0, outy0 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2);
    vstore4(ysf.s1357, 0, outy1 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2);
}

__kernel void loaduv(__global uchar8 const * const in,
                     __global float8 * out,
                     int out_offset)
{
  const int gid = get_global_id(0);
  const uchar8 inv = in[gid];

  // y = (x - 128) / 128
  const float8 outv  = (convert_float8(inv) - 128.f) * 0.0078125f;
  out[gid + out_offset / 8] = outv;
}

""").build()

'''

import os
import ctypes

from numpy.ctypeslib import ndpointer
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import cv2
from numba import jit, prange


def warp_coord(M, coord):
    if M.shape[0] == 3:
        x = (M[0, 0]*coord[0] + M[0, 1]*coord[1] + M[0, 2])/(M[2, 0]*coord[0] + M[2, 1]*coord[1] + M[2, 2])
        y = (M[1, 0]*coord[0] + M[1, 1]*coord[1] + M[1, 2])/(M[2, 0]*coord[0] + M[2, 1]*coord[1] + M[2, 2])
    else:
        x = M[0, 0]*coord[0] + M[0, 1]*coord[1] + M[0, 2]
        y = M[1, 0]*coord[0] + M[1, 1]*coord[1] + M[1, 2]

    warped_coord = np.array([x, y])
    return warped_coord


def warp_corners(mtx, corners):
    # corners = np.array([[w_lo, h_lo],
    #                       [w_hi, h_lo],
    #                       [w_hi, h_hi],
    #                       [w_lo, h_hi]])
    warped_corners = np.zeros((corners.shape[0], 2))
    for i, c in enumerate(corners):
        warped_corners[i] = warp_coord(mtx, c)
    return warped_corners


def ecef2geodetic(ecef, radians=False):
    """
    Convert ECEF coordinates to geodetic using ferrari's method
    """
    a = 6378137
    b = 6356752.3142
    esq = 6.69437999014 * 0.001
    e1sq = 6.73949674228 * 0.001

    # Save shape and export column
    ecef = np.atleast_1d(ecef)
    input_shape = ecef.shape
    ecef = np.atleast_2d(ecef)
    x, y, z = ecef[:, 0], ecef[:, 1], ecef[:, 2]

    ratio = 1.0 if radians else (180.0 / np.pi)

    # Conver from ECEF to geodetic using Ferrari's methods
    # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#Ferrari.27s_solution
    r = np.sqrt(x * x + y * y)
    Esq = a * a - b * b
    F = 54 * b * b * z * z
    G = r * r + (1 - esq) * z * z - esq * Esq
    C = (esq * esq * F * r * r) / (pow(G, 3))
    S = np.cbrt(1 + C + np.sqrt(C * C + 2 * C))
    P = F / (3 * pow((S + 1 / S + 1), 2) * G * G)
    Q = np.sqrt(1 + 2 * esq * esq * P)
    r_0 = -(P * esq * r) / (1 + Q) + np.sqrt(0.5 * a * a*(1 + 1.0 / Q) -
                                             P * (1 - esq) * z * z / (Q * (1 + Q)) - 0.5 * P * r * r)
    U = np.sqrt(pow((r - esq * r_0), 2) + z * z)
    V = np.sqrt(pow((r - esq * r_0), 2) + (1 - esq) * z * z)
    Z_0 = b * b * z / (a * V)
    h = U * (1 - b * b / (a * V))
    lat = ratio*np.arctan((z + e1sq * Z_0) / r)
    lon = ratio*np.arctan2(y, x)

    # stack the new columns and return to the original shape
    geodetic = np.column_stack((lat, lon, h))
    return geodetic.reshape(input_shape)


MAT_YUV2RGB = np.array([[1.1643, 0, 1.5958],
                        [1.1643, - 0.39173, - 0.81290],
                        [1.1643, 2.017, 0]]).T
MAT_RGB2YUV = np.linalg.inv(MAT_YUV2RGB)


def _yuv2rgb(new_img):
    img = new_img * 128
    img[:, :, :4] += 112
    pix = np.zeros((new_img.shape[0] * 2, new_img.shape[1] * 2, 3))
    pix[::2, ::2] = img[:, :, np.array([0, 4, 5])].dot(MAT_YUV2RGB)
    pix[::2, 1::2] = img[:, :, np.array([1, 4, 5])].dot(MAT_YUV2RGB)
    pix[1::2, ::2] = img[:, :, np.array([2, 4, 5])].dot(MAT_YUV2RGB)
    pix[1::2, 1::2] = img[:, :, np.array([3, 4, 5])].dot(MAT_YUV2RGB)
    return pix


def _rgb2yuv(pix):
    img = np.zeros((pix.shape[0] // 2, pix.shape[1] // 2, 6))
    img[:, :, [0, 4, 5]] = pix[::2, ::2].dot(MAT_RGB2YUV)
    img[:, :, [1, 4, 5]] = pix[::2, 1::2].dot(MAT_RGB2YUV)
    img[:, :, [2, 4, 5]] = pix[1::2, ::2].dot(MAT_RGB2YUV)
    img[:, :, [3, 4, 5]] = pix[1::2, 1::2].dot(MAT_RGB2YUV)

    img[:, :, :4] -= 112
    img = img / 128
    return img


@jit(nopython=True, parallel=True, nogil=True)
def yuv2rgb(new_img):
    pix = np.zeros((new_img.shape[0] * 2, new_img.shape[1] * 2, 3))

    for i in prange(new_img.shape[0]):
        for j in prange(new_img.shape[1]):
            for k in prange(4):
                Y_val = new_img[i, j, k] * 128 + 112
                U_val = new_img[i, j, 4] * 128
                V_val = new_img[i, j, 5] * 128

                B = 1.1643 * (Y_val) + 2.017 * (U_val)
                G = 1.1643 * (Y_val) - 0.81290 * (V_val) - 0.39173 * (U_val)
                R = 1.1643 * (Y_val) + 1.5958 * (V_val)

                p = (R, G, B)
                if k == 0:
                    pix[i * 2, j * 2] = p
                elif k == 1:
                    pix[i * 2, j * 2 + 1] = p
                elif k == 2:
                    pix[i * 2 + 1, j * 2] = p
                elif k == 3:
                    pix[i * 2 + 1, j * 2 + 1] = p
    return pix


@jit(nopython=True, parallel=True, nogil=True)
def rgb2yuv(pix):
    img = np.zeros((pix.shape[0] // 2, pix.shape[1] // 2, 6))

    for i in prange(img.shape[0]):
        for j in prange(img.shape[1]):
            U_val = 0
            V_val = 0
            for k in prange(4):
                if k == 0:
                    p = pix[i * 2, j * 2]
                elif k == 1:
                    p = pix[i * 2, j * 2 + 1]
                elif k == 2:
                    p = pix[i * 2 + 1, j * 2]
                elif k == 3:
                    p = pix[i * 2 + 1, j * 2 + 1]

                Y_val = 0.25681631 * p[0] + 0.50415484 * p[1] + 0.09791402 * p[2]
                U = -0.14824553 * p[0] + -0.29102007 * p[1] + 0.4392656 * p[2]
                V = 0.43927107 * p[0] + -0.36783273 * p[1] - 0.07143833 * p[2]

                img[i, j, k] = (Y_val - 112) / 128
                U_val += U
                V_val += V

            img[i, j, 4] = U_val / 4 / 128
            img[i, j, 5] = V_val / 4 / 128
    return img




class AdamOpt:

    def __init__(self, size, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, dtype=np.float32):
        self.exp_avg = np.zeros(size, dtype=dtype)
        self.exp_avg_sq = np.zeros(size, dtype=dtype)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.step = 0

    def update(self, grad):

        self.step += 1

        bias_correction1 = 1 - self.beta1 ** self.step
        bias_correction2 = 1 - self.beta2 ** self.step

        self.exp_avg = self.beta1 * self.exp_avg + (1 - self.beta1) * grad
        self.exp_avg_sq = self.beta2 * self.exp_avg_sq + (1 - self.beta2) * (grad ** 2)

        denom = (np.sqrt(self.exp_avg_sq) / np.sqrt(bias_correction2)) + self.eps

        step_size = self.lr / bias_correction1

        return step_size / denom * self.exp_avg
'''