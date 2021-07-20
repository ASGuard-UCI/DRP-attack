#!/usr/bin/env python
from __future__ import print_function
import math
import numpy as np

from selfdrive.car.car_helpers import interfaces  # pylint: disable=import-error
from selfdrive.controls.lib.vehicle_model import (
    VehicleModel,
)  # pylint: disable=import-error
from selfdrive.controls.lib.pathplanner import (
    PathPlanner,
)  # pylint: disable=import-error

from car_motion_attack.polyfuzz.utils.vehicle_control import VehicleControl, VehicleControlDBM
from car_motion_attack.polyfuzz.utils.mock_latcontrol import MockLatControl
from car_motion_attack.polyfuzz.utils.parse_model_output import parse_model_output


def steer_angle_to_wheel_radian(steer_angle, steer_ratio):
    return math.radians(steer_angle) / steer_ratio

class LaneLine:
    def __init__(self, poly, prob, points=None):
        self.poly = poly
        self.points = points
        self.prob = prob

class ModelOutput:
    def __init__(self, l_poly, r_poly, p_poly, l_prob, r_prob):
        self.leftLane = LaneLine(l_poly, l_prob)
        self.rightLane = LaneLine(r_poly, r_prob)
        self.path = LaneLine(p_poly, 1.)


class PolyFuzz(object):
    def __init__(self, v0=20.0):
        interface_cls = interfaces["mock"]  # TOYOTA RAV4 HYBRID 2017
        self.CP = interface_cls[0].get_params("mock", None)  # Car Parameters
        self.VM = VehicleModel(self.CP)  # Vehicle Model
        # self.CI = interface_cls(self.CP, False)     # Car Interface
        # # Set dummy car state
        # self.CI.prev_speed = v0
        # self.CI.speed = v0
        # self.car_state = self.CI.update(None)
        self.v_ego = v0
        self.angle_steers = 0.0
        self.PP = PathPlanner(self.CP)
        self.angle_steers_des_mpc = 0.0

    def run(self, l_poly, r_poly, p_poly, l_prob, r_prob):

        md = ModelOutput(l_poly, r_poly, p_poly, l_prob, r_prob)

        self.PP.update_poly(
            self.CP,
            self.VM,
            self.v_ego,
            self.angle_steers,
            md
        )
        plan_valid = self.PP.solution_invalid_cnt < 2
        cost = self.PP.mpc_solution[0].cost
        # Desired wheel angle change
        angle = math.degrees(self.PP.mpc_solution[0].delta[1])
        # Desired steering angle change
        self.angle_steers_des_mpc = self.PP.angle_steers_des_mpc
        return plan_valid, cost, angle

    def update_state(self, v_ego, angle_steers):
        self.v_ego = v_ego
        self.angle_steers = angle_steers


class VehicleState(object):
    def __init__(self, v0=20.0, freq=100, yaw=0.0, model=VehicleControl):
        interface_cls = interfaces["mock"][0]  # TOYOTA RAV4 HYBRID 2017
        self.CP = interface_cls.get_params("mock", None)  # Car Parameters
        #self.CP.wheelbase = 3
        #self.CP.steerRatio = 20
        self.VM = VehicleModel(self.CP)  # Vehicle Model
        self.v_ego = v0
        self.VC = model(
            velocity=self.v_ego, wheelbase=self.CP.wheelbase, yaw=yaw
        )
        #self.LaC = MockLatControl(self.CP)
        #self.LaC.reset()
        self.steer_angle_measure = 0.0

        self.current_steering_angle = 0
        self.desired_steer_angle_mpc = 0
        self.wheel_rad = 0.0
        self.sat_flag = False
        self.steer = 0.0
        self.duration = 1.0 / freq

    def apply_plan(self, desired_steer_angle_mpc, duration=None):
        if duration is None:
            duration = self.duration

        self.desired_steer_angle_mpc = desired_steer_angle_mpc
        self.wheel_rad = steer_angle_to_wheel_radian(
            desired_steer_angle_mpc, self.CP.steerRatio
        )
        # self.wheel_rad_delta = steer_angle_to_wheel_radian(self.current_steering_angle - self.desired_steer_angle_mpc, self.CP.steerRatio)

        state = self.VC.get_state(self.wheel_rad, duration)
        return state

    def update_velocity(self, v):
        self.v_ego = v
        self.VC.update_velocity(v)

    def update_steer(self, steer):
        self.current_steering_angle = steer


def main():
    v_ego = 20.0
    angle_steers = 0.0
    modelout = np.load("./testdata/yuv6c_50f_highwayramp_suv_modelout.npy")
    p_std, l_std, r_std, l_prob, r_prob, p_poly, l_poly, r_poly = parse_model_output(
        modelout[0]
    )  # pylint: disable=unused-variable
    PF = PolyFuzz()
    VS = VehicleState()
    valid, cost, angle = PF.run(
        l_poly, r_poly, p_poly, l_prob, r_prob
    )  # pylint: disable=unused-variable
    angle_steers_des = PF.angle_steers_des_mpc
    print("Desired wheel angle and radian:", angle, math.radians(angle))
    x = []
    y = []
    yaw = []
    for _ in range(5):
        state = VS.apply_plan(angle_steers_des)
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
    angle_steers = angle_steers_des
    PF.update_state(v_ego, angle_steers)

    # max_angle = 0.
    # max_l_poly = None
    # max_r_poly = None
    # max_d = 0.
    # max_cost = 0.
    # for d in np.arange(-3.8, 3.8, 0.1):
    #     l_poly_eval = np.array(l_poly, copy=True)
    #     r_poly_eval = np.array(r_poly, copy=True)
    #     l_poly_eval[3] += d
    #     r_poly_eval[3] += d
    #     PF = PolyFuzz()
    #     valid, cost, angle = PF.run(l_poly_eval, r_poly_eval,
    #                                 p_poly, l_prob, r_prob)
    #     if valid and abs(angle) > abs(max_angle):
    #         max_angle = angle
    #         max_l_poly = np.array(l_poly_eval, copy=True)
    #         max_r_poly = np.array(r_poly_eval, copy=True)
    #         max_d = d
    #         max_cost = cost
    #     del PF
    # print()
    # print("Best d:", max_d)
    # print("Best cost:", max_cost)
    # print("Best angle:", max_angle)


if __name__ == "__main__":
    main()
