import numpy as np
import pandas as pd
import cv2
from logging import getLogger
import pymap3d as pm


import comma2k19.orientation as orient

from car_motion_attack.polyfuzz.polyfuzz import VehicleState
from car_motion_attack.utils import ecef2geodetic
from car_motion_attack.polyfuzz.utils.vehicle_control import VehicleControl, VehicleControlDBM

logger = getLogger(__name__)


def load_transform_matrix(path, start_time):
    # openpilot tools 
    from tools.lib.logreader import LogReader
    raw_log = LogReader(path)
    ext_mat = None
    for l in raw_log:
        try:
            if l.which() == 'liveCalibration':
                t = l.logMonoTime * 1e-9
                if ext_mat is None:
                    ext_mat = l.liveCalibration.extrinsicMatrix

                if t > start_time:
                    break

                ext_mat = l.liveCalibration.extrinsicMatrix
        except:
            continue

    if ext_mat is None:
        raise Exception(f'no transform matrix for time={start_time}')

    ext = np.array(ext_mat).reshape((3, 4))
    eon = np.array([[910.0, 0.0, 582.0],
                    [0.0, 910.0, 437.0],
                    [0.0,   0.0,   1.0]])
    gm = np.array([[0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                   [-1.09890110e-03, 0.00000000e+00, 2.81318681e-01],
                   [-1.84808520e-20, 9.00738606e-04, -4.28751576e-02]])
    camera_frame_from_ground = np.dot(eon, ext)[:, [0, 1, 3]]

    trns = np.dot(camera_frame_from_ground, gm)
    return trns


def _conv_ecefs2enu(ecefs):
    pos = []
    lat0, lon0, h0 = pm.ecef2geodetic(
        ecefs[0][0], ecefs[0][1], ecefs[0][2], ell=None, deg=True
    )
    for i, ecef in enumerate(ecefs):
        e, n, u = pm.ecef2enu(ecef[0], ecef[1], ecef[2], lat0, lon0, h0)
        pos.append([e, n])

    pos = np.array(pos)
    pos -= pos[0]

    d = 1
    r = np.sqrt((pos[d] ** 2).sum())
    _cos = pos[d][0] / r
    _sin = -pos[d][1] / r

    mat_rotate = np.array([[_cos, -_sin], [_sin, _cos]])

    pos = np.dot(pos, mat_rotate.T)

    return pos


def load_sensor_data(path, offset=0):
    t = np.load(path + "global_pose/frame_times")
    map_vals = {"t": t[offset:]}

    speed = np.load(path + "processed_log/CAN/speed/value")[:, 0]
    t_speed = np.load(path + "processed_log/CAN/speed/t")
    all_speed = np.interp(t, t_speed, speed)[offset:]

    map_vals.update({"speed": all_speed, "mph": all_speed * 2.237})

    steering_angle = np.load(path + "processed_log/CAN/steering_angle/value")
    t_steering_angle = np.load(path + "processed_log/CAN/steering_angle/t")
    all_steering_angle = np.interp(t, t_steering_angle, steering_angle)

    map_vals.update({"steering_angle": all_steering_angle[offset:]})

    ecefs = np.load(path + "global_pose/frame_positions")[offset:]
    pos = _conv_ecefs2enu(ecefs)

    map_vals.update({"lateral_shift": pos[:, 1]})
    map_vals.update({"longitude_shift": pos[:, 0]})

    orientations = np.load(path + "global_pose/frame_orientations")

    yaw = -orient.ned_euler_from_ecef(ecefs[0], orient.euler_from_quat(orientations))[
        :, 2
    ]
    yaw = yaw[offset:]
    yaw = yaw % (2 * np.pi)

    map_vals.update({"yaw": yaw - yaw[0]})

    df = pd.DataFrame(map_vals)
    df["t_diff"] = df["t"].diff()
    df.loc[0, "t_diff"] = 0
    df["distance"] = df["t_diff"] * df["speed"]

    return df


def load_sensor_data_bicycle(path):
    t = np.load(path + "global_pose/frame_times")
    map_vals = {"t": t}

    speed = np.load(path + "processed_log/CAN/speed/value")[:, 0]
    t_speed = np.load(path + "processed_log/CAN/speed/t")
    all_speed = np.interp(t, t_speed, speed)

    map_vals.update({"speed": all_speed, "mph": all_speed * 2.237})

    steering_angle = np.load(path + "processed_log/CAN/steering_angle/value")
    t_steering_angle = np.load(path + "processed_log/CAN/steering_angle/t")
    all_steering_angle = np.interp(t, t_steering_angle, steering_angle)

    map_vals.update({"steering_angle": all_steering_angle})

    df = pd.DataFrame(map_vals)
    df["t_diff"] = df["t"].diff()
    df.loc[0, "t_diff"] = 0
    df["distance"] = df["t_diff"] * df["speed"]

    vehicle_state = VehicleState()
    list_lateral_shift = [0]
    list_longitude_shift = [0]
    list_yaw = [0]
    for i in range(df.shape[0]):  # loop on 20Hz
        # update vehicle state
        v_ego = df.loc[i, "speed"]

        vehicle_state.update_velocity(v_ego)
        vehicle_state.angle_steers = df.loc[i, "steering_angle"]

        # update steering angle
        for _ in range(5):  # loop on 100Hz
            state = vehicle_state.apply_plan(df.loc[i, "steering_angle"])
        list_lateral_shift.append(state.y)
        list_longitude_shift.append(state.x)
        list_yaw.append(state.yaw)

    df["lateral_shift"] = list_lateral_shift[:-1]
    df["longitude_shift"] = list_longitude_shift[:-1]
    df["yaw"] = list_yaw[:-1]

    return df


def load_sensor_data_bicycleDBM(path):
    t = np.load(path + "global_pose/frame_times")
    map_vals = {"t": t}

    speed = np.load(path + "processed_log/CAN/speed/value")[:, 0]
    t_speed = np.load(path + "processed_log/CAN/speed/t")
    all_speed = np.interp(t, t_speed, speed)

    map_vals.update({"speed": all_speed, "mph": all_speed * 2.237})

    steering_angle = np.load(path + "processed_log/CAN/steering_angle/value")
    t_steering_angle = np.load(path + "processed_log/CAN/steering_angle/t")
    all_steering_angle = np.interp(t, t_steering_angle, steering_angle)

    map_vals.update({"steering_angle": all_steering_angle})

    df = pd.DataFrame(map_vals)
    df["t_diff"] = df["t"].diff()
    df.loc[0, "t_diff"] = 0
    df["distance"] = df["t_diff"] * df["speed"]

    vehicle_state = VehicleState(model=VehicleControlDBM)
    list_lateral_shift = [0]
    list_longitude_shift = [0]
    list_yaw = [0]
    for i in range(df.shape[0]):  # loop on 20Hz
        # update vehicle state
        v_ego = df.loc[i, "speed"]

        vehicle_state.update_velocity(v_ego)
        vehicle_state.angle_steers = df.loc[i, "steering_angle"]

        # update steering angle
        for _ in range(5):  # loop on 100Hz
            state = vehicle_state.apply_plan(df.loc[i, "steering_angle"])
        list_lateral_shift.append(state.y)
        list_longitude_shift.append(state.x)
        list_yaw.append(state.yaw)

    df["lateral_shift"] = list_lateral_shift[:-1]
    df["longitude_shift"] = list_longitude_shift[:-1]
    df["yaw"] = list_yaw[:-1]

    return df


if __name__ == "__main__":
    load_sensor_data(
        "data/straight_scenarios/b0c9d2329ad1606b|2018-08-03--10-35-16/4/"
    ).head(10)
