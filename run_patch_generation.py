import sys
import pickle
import os
import gc
import json
from logging import getLogger


import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tqdm import tqdm

try:
    sys.path.append("/home/takamisato/openpilot/")
except:
    raise

from car_motion_attack.attack import CarMotionAttack
from car_motion_attack.replay_bicycle import ReplayBicycle
from car_motion_attack.load_sensor_data import load_sensor_data  # , load_transform_matrix

logger = getLogger(None)

EVAL_FRAMES = 50 #2.5 sec

def main(data_path='',
         n_epoch=10000,
         n_frames=20,
         scale=5,
         base_color=0.38,
         starting_meters=45,
         patch_lateral_shift=0,
         result_dir='./result/',
         left_lane_pos=4,
         right_lane_pos=36,
         left_solid=False,
         right_solid=False,
         src_corners=None,
         target_deviation=0.5,
         is_attack_to_rigth=True,
         patch_width=45,
         patch_length=300,
         frame_offset=0,
         l2_weight=0.01
         ):

    df_sensors = load_sensor_data(data_path, offset=frame_offset).head(n_frames + 1)
    if not os.path.exists(data_path + "imgs/"):
        os.mkdir(data_path + "imgs/")
        vc = cv2.VideoCapture(data_path + "video.hevc")
        i = 0
        while True:
            rval, frame = vc.read()
            if not rval:
                break
            cv2.imwrite(data_path + f"imgs/{i}.png", frame)
            i += 1

    #roi_mat = load_transform_matrix(data_path + "raw_log.bz2", start_time=df_sensors.loc[0, "t"])
    roi_mat = np.load(data_path + "trns.npy")

    list_bgr_img = [cv2.imread(data_path + f"imgs/{i}.png") for i in range(frame_offset, frame_offset + n_frames + 1)]
    global_bev_mask = np.random.random((patch_length * scale, patch_width * scale)) > 0

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=False)

    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        cm = ReplayBicycle(
            sess, list_bgr_img, df_sensors, global_bev_mask, roi_mat, scale=scale
        ).run(start_steering_angle=None)
        df_sensors["lateral_shift_openpilot"] = [0] + cm.list_total_lateral_shift[:-1]
        df_sensors["yaw_openpilot"] = [0] + cm.list_yaw[:-1]

        cma = CarMotionAttack(
            sess,
            list_bgr_img,
            df_sensors,
            global_bev_mask,
            base_color,
            roi_mat,
            scale=scale,
            n_epoch=n_epoch,
            result_dir=result_dir,
            left_lane_pos=left_lane_pos,
            right_lane_pos=right_lane_pos,
            src_corners=src_corners,
            is_attack_to_rigth=is_attack_to_rigth,
            target_deviation=target_deviation,
            l2_weight=l2_weight
        )
        cma.run(
            starting_meters=starting_meters,
            lateral_shift=patch_lateral_shift,
            starting_steering_angle=cm.list_desired_steering_angle[0],
            # starting_patch_dir=START_DIR,
            # starting_patch_epoch=START_DIR_EPOCH,
        )
        last_epoch = cma.last_epoch
        par = cma.perturbable_area_ratio
        del cma, list_bgr_img
        gc.collect()

    result = {"data_path": data_path,
              "n_epoch": n_epoch,
              "n_frames": n_frames,
              "scale": scale,
              "base_color": base_color,
              "starting_meters": starting_meters,
              "patch_lateral_shift": patch_lateral_shift,
              "result_dir": result_dir,
              "left_lane_pos": left_lane_pos,
              "right_lane_pos": right_lane_pos,
              "src_corners": src_corners,
              "target_deviation": target_deviation,
              "is_attack_to_rigth": is_attack_to_rigth,
              "perturbable_area_ratio": par,
              'last_epoch': last_epoch}
    with open(result_dir + "result.json", "w") as f:
        f.write(json.dumps(result))

    df_sensors = load_sensor_data(data_path, offset=frame_offset).head(EVAL_FRAMES)
    list_bgr_img = [cv2.imread(data_path + f"imgs/{i}.png") for i in range(frame_offset, frame_offset + EVAL_FRAMES)]

    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        cm = ReplayBicycle(
            sess, list_bgr_img, df_sensors, global_bev_mask, roi_mat, scale=scale
        ).run(start_steering_angle=None)
        df_sensors["lateral_shift_openpilot"] = [0] + cm.list_total_lateral_shift[:-1]
        df_sensors["yaw_openpilot"] = [0] + cm.list_yaw[:-1]
        cma_rep = CarMotionAttack(
            sess,
            list_bgr_img,
            df_sensors,
            global_bev_mask,
            base_color,
            roi_mat,
            scale=scale,
            n_epoch=n_epoch,
            result_dir=result_dir,
            left_lane_pos=left_lane_pos,
            right_lane_pos=right_lane_pos,
            src_corners=src_corners,
            is_attack_to_rigth=is_attack_to_rigth,
            target_deviation=target_deviation
        )

        cma_rep.replay(
            epoch=last_epoch,
            starting_meters=starting_meters,
            lateral_shift=patch_lateral_shift,
            starting_steering_angle=cm.list_desired_steering_angle[0],
        )


if __name__ == "__main__":

    from logging import StreamHandler, Formatter, FileHandler
    config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config = json.loads(f.read())

    os.makedirs(config["result_dir"], exist_ok=True)
    log_fmt = Formatter(
        "%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s "
    )

    handler = StreamHandler()
    handler.setLevel("INFO")
    handler.setFormatter(log_fmt)
    logger.setLevel("INFO")
    logger.addHandler(handler)

    handler = FileHandler(
        config["result_dir"] + os.path.basename(os.path.abspath(__file__)) + ".log", "w"
    )
    handler.setLevel("DEBUG")
    handler.setFormatter(log_fmt)
    handler.setLevel("DEBUG")
    logger.addHandler(handler)

    logger.info("start")
    main(**config)
    logger.info("end")
