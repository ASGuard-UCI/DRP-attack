import numpy as np

MODEL_PATH_DISTANCE = 192
POLYFIT_DEGREE = 4
SPEED_PERCENTILES = 10


path_start = 0
left_start = MODEL_PATH_DISTANCE * 2
right_start = MODEL_PATH_DISTANCE * 2 + MODEL_PATH_DISTANCE*2 + 1
lead_start = MODEL_PATH_DISTANCE * 2 + (MODEL_PATH_DISTANCE*2 + 1) * 2
num_pts = MODEL_PATH_DISTANCE

def softplus(x):
    return np.log1p(np.exp(x))
    

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def vander():
    mat = np.zeros((MODEL_PATH_DISTANCE, POLYFIT_DEGREE-1))
    for i in range(MODEL_PATH_DISTANCE):
        for j in range(POLYFIT_DEGREE-1):
            mat[i,j] = pow(i, POLYFIT_DEGREE-j-1)
    return mat
    
mat_vander = vander()

def poly_fit(pts, stds):
    poly = np.zeros((4,))
    lhs = mat_vander / stds[:, None]

    rhs = (pts - pts[0]) / stds


    poly_3, res, dim, sing = np.linalg.lstsq(lhs, rhs, rcond=None)

    poly[:3] = poly_3
    poly[3] = pts[0]
    return poly, res, dim, sing


def postprocess(model_out):
    path_pts = model_out[path_start: path_start + num_pts]
    left_pts = model_out[left_start: left_start + num_pts] + 1.8
    right_pts = model_out[right_start: right_start + num_pts] - 1.8

    path_stds = softplus(model_out[path_start + num_pts: path_start + num_pts + num_pts])
    left_stds = softplus(model_out[left_start + num_pts: left_start + num_pts + num_pts])
    right_stds = softplus(model_out[right_start + num_pts: right_start + num_pts + num_pts])

    #path_std = softplus(model_out[path_start + num_pts+num_pts // 4])
    #left_std = softplus(model_out[left_start + num_pts+num_pts // 4])
    #right_std = softplus(model_out[right_start + num_pts+num_pts // 4])

    #path_prob = 1.
    left_prob = sigmoid(model_out[left_start + num_pts * 2])
    right_prob = sigmoid(model_out[right_start + num_pts * 2])

    path_poly, path_res, _, _ = poly_fit(path_pts, path_stds)
    left_poly, left_res, _, _ = poly_fit(left_pts, left_stds)
    right_poly, right_res, _, _ = poly_fit(right_pts, right_stds)


    return path_poly, left_poly, right_poly, left_prob, right_prob
