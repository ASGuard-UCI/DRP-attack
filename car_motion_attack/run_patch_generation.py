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
from car_motion_attack.load_sensor_data import load_sensor_data

logger = getLogger(None)


def get_result_replay(path, epoch=10000):
    res = []
    lateral_shift = total_lateral_shift = lateral_shift_openpilot = 0
    with open(path) as f:
        # for line in f:
        #    if f'start {epoch}' in line:
        #        break
        for line in f:
            #total_lateral_shift = 0
            if f'start {epoch + 1}' in line:
                break
            elif ': yaw_diff:' in line:
                paaa = line
                lateral_shift = float(line.strip().split(':')[-3].split(',')[0])
                total_lateral_shift = float(line.strip().split(':')[-2].split(',')[0])
                lateral_shift_openpilot = float(line.strip().split(':')[-1].split(',')[0])
            elif ': valid:' in line:
                # print(line.strip().split(':'))
                try:
                    cost = float(line.strip().split(':')[-2].split(',')[0])
                    line = f.readline()

                    tmp = line.strip().split(':')
                    desired_steering_angle = float(tmp[-2].split(',')[0])
                    current_steering_angle = float(tmp[-1])
                except:
                    pass
            elif '[DEBUG][update_perspective] enter' in line or '[update_trajectory] exit' in line:
                res.append((cost, desired_steering_angle, current_steering_angle,
                            lateral_shift, total_lateral_shift, lateral_shift_openpilot))
    df = pd.DataFrame(res, columns=['cost', 'desired_steering_angle', 'current_steering_angle',
                                    'lateral_shift', 'total_lateral_shift', 'lateral_shift_openpilot'])
    df.loc[0, ['lateral_shift', 'total_lateral_shift', 'lateral_shift_openpilot']] = 0
    return df


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

    list_bgr_img = [cv2.imread(data_path + f"imgs/{i}.png") for i in range(frame_offset, frame_offset + n_frames + 1)]
    global_bev_mask = np.random.random((patch_length * scale, patch_width * scale)) > 0

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=False)

    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        cm = ReplayBicycle(
            sess, list_bgr_img, df_sensors, global_bev_mask, scale=scale
        ).run(start_steering_angle=None)
        df_sensors["lateral_shift_openpilot"] = [0] + cm.list_total_lateral_shift[:-1]
        df_sensors["yaw_openpilot"] = [0] + cm.list_yaw[:-1]

        cma = CarMotionAttack(
            sess,
            list_bgr_img,
            df_sensors,
            global_bev_mask,
            base_color=base_color,
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

    df_sensors = load_sensor_data(data_path, offset=frame_offset).head(46 + 1)
    list_bgr_img = [cv2.imread(data_path + f"imgs/{i}.png") for i in range(frame_offset, frame_offset + 46 + 1)]

    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        cm = ReplayBicycle(
            sess, list_bgr_img, df_sensors, global_bev_mask, scale=scale
        ).run(start_steering_angle=None)
        df_sensors["lateral_shift_openpilot"] = [0] + cm.list_total_lateral_shift[:-1]
        df_sensors["yaw_openpilot"] = [0] + cm.list_yaw[:-1]
        cma_rep = CarMotionAttack(
            sess,
            list_bgr_img,
            df_sensors,
            global_bev_mask,
            base_color=base_color,
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
        df_result = get_result_replay(result_dir + os.path.basename(os.path.abspath(__file__)) + ".log",
                                      last_epoch)
        df_result.to_csv(result_dir + 'result.csv')


if __name__ == "__main__":

    from logging import StreamHandler, Formatter, FileHandler
    config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config = json.loads(f.read())
        #config['l2_weight'] = 0.01
        config['n_epoch'] = 1000

    os.makedirs(config["result_dir"] + "replay/", exist_ok=True)
    log_fmt = Formatter(
        "%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s "
    )

    handler = StreamHandler()
    handler.setLevel("INFO")
    handler.setFormatter(log_fmt)
    logger.setLevel("INFO")
    logger.addHandler(handler)

    handler = FileHandler(
        config["result_dir"] + os.path.basename(os.path.abspath(__file__)) + ".log", "a"
    )
    handler.setLevel("DEBUG")
    handler.setFormatter(log_fmt)
    handler.setLevel("DEBUG")
    logger.addHandler(handler)

    logger.info("start")
    main(**config)
    logger.info("end")
