import math
from selfdrive.controls.lib.lane_planner import model_polyfit, compute_path_pinv

def softmax(inputs):
  max_val = max(inputs)
  denominator = 0.
  outputs = [0.]*len(inputs)
  for i in range(len(inputs)):
    v_exp = math.exp(inputs[i] - max_val)
    denominator += v_exp
    outputs[i] = v_exp
  inv_denominator = 1./denominator
  for i in range(len(inputs)):
    outputs[i] *= inv_denominator
  return outputs

PATH_START = 0
PATH_END = 50
LEFT_START = 51
LEFT_END = LEFT_START+50
RIGHT_START = 51+53
RIGHT_END = RIGHT_START+50

path_pinv = compute_path_pinv()

def parse_model_output(output):
  p_std = math.sqrt(2.)/output[PATH_START+50]
  l_std = math.sqrt(2.)/output[LEFT_START+50]
  r_std = math.sqrt(2.)/output[RIGHT_START+50]
  l_prob = softmax(output[LEFT_END+1:LEFT_END+1+2])[0]
  r_prob = softmax(output[RIGHT_END+1:RIGHT_END+1+2])[0]
  path = output[PATH_START:PATH_END]
  left = output[LEFT_START:LEFT_END] + 1.8
  right = output[RIGHT_START:RIGHT_END] - 1.8
  p_poly = model_polyfit(path, path_pinv)  # predicted path
  l_poly = model_polyfit(left, path_pinv)  # left line
  r_poly = model_polyfit(right, path_pinv)  # right line
  return p_std, l_std, r_std, l_prob, r_prob, p_poly, l_poly, r_poly

def parse_model_output_raw(output):
  p_std = math.sqrt(2.)/output[PATH_START+50]
  l_std = math.sqrt(2.)/output[LEFT_START+50]
  r_std = math.sqrt(2.)/output[RIGHT_START+50]
  l_prob = softmax(output[LEFT_END+1:LEFT_END+1+2])[0]
  r_prob = softmax(output[RIGHT_END+1:RIGHT_END+1+2])[0]
  path = output[PATH_START:PATH_END]
  left = output[LEFT_START:LEFT_END] + 1.8
  right = output[RIGHT_START:RIGHT_END] - 1.8
  return path, left, right
