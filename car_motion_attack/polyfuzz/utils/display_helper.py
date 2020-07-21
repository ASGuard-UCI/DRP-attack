import numpy as np
import cv2
import matplotlib.pyplot as plt


def display(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()


def display_rgb(img_rgb):
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.imshow(img_rgb)
    plt.show()


def make_lut_u():
    return np.array([[[i, 255-i, 0] for i in range(256)]], dtype=np.uint8)


def make_lut_v():
    return np.array([[[0, 255-i, i] for i in range(256)]], dtype=np.uint8)


def y_to_bgr(y):
    return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)


def u_to_bgr(u):
    lut_u = make_lut_u()
    u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
    return cv2.LUT(u, lut_u)


def v_to_bgr(v):
    lut_v = make_lut_v()
    v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    return cv2.LUT(v, lut_v)


def display_y(y):
    y = y_to_bgr(y)
    display(y)


def display_u(u):
    u = u_to_bgr(u)
    display(u)


def display_v(v):
    v = v_to_bgr(v)
    display(v)


def display_yuv(y, u, v):
    y = y_to_bgr(y)
    u = u_to_bgr(u)
    v = v_to_bgr(v)
    yuv_bgr = np.vstack([y, u, v])
    display(yuv_bgr)


def display_yuv_comb(img_yuv):
    y, u, v = cv2.split(img_yuv)
    display_yuv(y, u, v)


def yuv_to_rgb(y, u, v):
    y = y_to_bgr(y)
    u = u_to_bgr(u)
    v = v_to_bgr(v)
    yuv_bgr = np.vstack([y, u, v])
    return yuv_bgr


def display_modelin(yuv6c):
    # yuv6c is a flat numpy array for the model input
    net_in = [ x * 128 + 128 for x in yuv6c ]
    chls = []
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    for c in range(6):
        chls.append(np.array(net_in[c*80*160:c*80*160+80*160], dtype=np.uint8).reshape(80, 160))
        if c < 4:
            ch = y_to_bgr(chls[c])
        elif c == 4:
            ch = u_to_bgr(chls[c])
        else:
            ch = v_to_bgr(chls[c])
        axs[c%2][c//2].imshow(ch)
    plt.tight_layout()
    plt.show()