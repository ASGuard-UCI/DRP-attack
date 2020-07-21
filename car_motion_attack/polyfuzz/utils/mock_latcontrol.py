from selfdrive.controls.lib.pid import PIController
from common.numpy_fast import interp
from cereal import car

_DT = 0.01    # 100Hz
_DT_MPC = 0.05  # 20Hz


def get_steer_max(CP, v_ego):
    steer_max = interp(v_ego, CP.steerMaxBP, CP.steerMaxV)
#     print("SteerMaxBP:", CP.steerMaxBP)
#     print("SteerMaxV:", CP.steerMaxV)
#     print("v_ego:", v_ego)
#     print("Steer Max:", steer_max)
    return steer_max


class MockLatControl(object):
    def __init__(self, CP):
        self.pid = PIController((CP.steerKpBP, CP.steerKpV),
                                (CP.steerKiBP, CP.steerKiV),
                                k_f=CP.steerKf, pos_limit=1.0)
        self.angle_steers_des = 0.

    def reset(self):
        self.pid.reset()

    def update(self, v_ego, angle_steers, CP, VM, angle_steers_des):
        if v_ego < 0.3:
            output_steer = 0.0
            print("Speed too low, resetting PID")
            self.pid.reset()
        else:
            # get from MPC/PathPlanner
            self.angle_steers_des = angle_steers_des
            steers_max = get_steer_max(CP, v_ego)
#             print(steers_max)
            self.pid.pos_limit = steers_max
            self.pid.neg_limit = -steers_max
            # feedforward desired angle
            steer_feedforward = self.angle_steers_des
            if CP.steerControlType == car.CarParams.SteerControlType.torque:
                # proportional to realigning tire momentum (~ lateral accel)
                steer_feedforward *= v_ego**2
            deadzone = 0.0
            output_steer = self.pid.update(self.angle_steers_des, angle_steers,
                                           check_saturation=(v_ego > 10),
                                           override=False,
                                           feedforward=steer_feedforward,
                                           speed=v_ego, deadzone=deadzone)

        self.sat_flag = self.pid.saturated
        return output_steer, float(self.angle_steers_des)
