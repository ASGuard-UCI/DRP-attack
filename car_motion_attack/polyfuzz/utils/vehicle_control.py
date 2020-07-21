from __future__ import print_function
import math

"""
Vehicle Kinematics Differential Equations
  x' = v * cos(yaw)
  y' = v * sin(yaw)
  yaw' = v / L * tan(steer)
  v' = a
"""


def xdot(v, yaw):
    return v * math.cos(yaw)


def ydot(v, yaw):
    return v * math.sin(yaw)


def yawdot(v, delta, L):
    return v / L * math.tan(delta)


def vdot(a):
    return 0   # Assume Acceleration is zero. i.e. Constant Velocity


class State:
    """
    State [Global_x, Global_y, Yaw(heading), Velocity]
        x: Global x         [m]
        y: Global y         [m]
        yaw: Heading angle  [rad]
        v: Velocity         [m/s]
    """
    def __init__(self, x=0., y=0., v=20., yaw=-math.pi/2):
        self.x = x
        self.y = y
        self.v = v
        self.yaw = yaw


class VehicleControl(object):
    def __init__(self, x=0., y=0., velocity=0., yaw=0., wheelbase=2.65):
        self.L = wheelbase
        self.state = State(x=x, y=y, v=velocity, yaw=yaw)

    def get_state(self, steer, duration):
        h = duration
        x1 = xdot(self.state.v, self.state.yaw)
        y1 = ydot(self.state.v, self.state.yaw)
        yaw1 = yawdot(self.state.v, steer, self.L)
        v1 = vdot(self.state.v)

        x2 = xdot(self.state.v + 0.5*h*v1, self.state.yaw + 0.5*h*yaw1)
        y2 = ydot(self.state.v + 0.5*h*v1, self.state.yaw + 0.5*h*yaw1)
        yaw2 = yawdot(self.state.v + 0.5*h*v1, steer, self.L)
        v2 = vdot(self.state.v + 0.5*h*v1)

        x3 = xdot(self.state.v + 0.5*h*v2, self.state.yaw + 0.5*h*yaw2)
        y3 = ydot(self.state.v + 0.5*h*v2, self.state.yaw + 0.5*h*yaw2)
        yaw3 = yawdot(self.state.v + 0.5*h*v2, steer, self.L)
        v3 = vdot(self.state.v + 0.5*h*v2)

        x4 = xdot(self.state.v + h*v3, self.state.yaw + h*yaw3)
        y4 = ydot(self.state.v + h*v3, self.state.yaw + h*yaw3)
        yaw4 = yawdot(self.state.v + h*v3, steer, self.L)
        v4 = vdot(self.state.v + h*v3)

        dx = h*(x1 + 2*x2 + 2*x3 + x4) / 6
        dy = h*(y1 + 2*y2 + 2*y3 + y4) / 6
        dyaw = h*(yaw1 + 2*yaw2 + 2*yaw3 + yaw4) / 6
        dv = h*(v1 + 2*v2 + 2*v3 + v4) / 6

        self.state.x += dx
        self.state.y += dy
        self.state.yaw += dyaw
        self.state.v += dv
        return self.state

    def update_velocity(self, v):
        self.state.v = v


#L = 3#2.6  # [m]
#Lr = L / 2.0  # [m]
#Lf = L - Lr
aaa = 1#0.5#0.5
Cf = 108873 * aaa #1600.0 * 2.0 * aaa # N/rad
Cr = 108873 * aaa#1700.0 * 2.0 * aaa # N/rad
#Iz = 2250.0  # kg/m2
m = 2249.0  # kg
Iz = m / 0.0592 # kg/m2


class State2:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, vx=0.01, vy=0.0, omega=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.omega = omega
"""
class VehicleControl(object):
    def __init__(self, x=0., y=0., velocity=0., yaw=0., wheelbase=2.65):
        self.L = wheelbase
        self.state = State2(x=x, y=y, vx=velocity, yaw=yaw)

    def get_state(self, steer, duration):

        s = self.state
        if s.vx < 1:
            return self.state

        delta = steer
        dt = duration
        Lr = self.L / 2.0  # [m]
        Lf = self.L - Lr

        a = 0
        s.x = s.x + s.vx * math.cos(s.yaw) * dt - s.vy * math.sin(s.yaw) * dt
        s.y = s.y + s.vx * math.sin(s.yaw) * dt + s.vy * math.cos(s.yaw) * dt
        s.yaw = s.yaw + s.omega * dt
        Ffy = -Cf * math.atan2(((s.vy + Lf * s.omega) / s.vx - delta), 1.0)
        Fry = -Cr * math.atan2((s.vy - Lr * s.omega) / s.vx, 1.0)
        s.vx = s.vx + (a - Ffy * math.sin(delta) / m + s.vy * s.omega) * dt
        s.vy = s.vy + (Fry / m + Ffy * math.cos(delta) / m - s.vx * s.omega) * dt
        s.omega = s.omega + (Ffy * Lf * math.cos(delta) - Fry * Lr) / Iz * dt
        self.state = s
        return self.state

    def update_velocity(self, v):
        self.state.vx = v
"""
class VehicleControlDBM(object):
    def __init__(self, x=0., y=0., velocity=0., yaw=0., wheelbase=2.65):
        self.L = wheelbase
        self.state = State2(x=x, y=y, vx=velocity, yaw=yaw)

    def get_state(self, steer, duration):

        s = self.state
        if s.vx < 1:
            return self.state

        delta = steer
        dt = duration
        Lr = self.L / 2.0  # [m]
        Lf = self.L - Lr

        a = 0
        s.x = s.x + s.vx * math.cos(s.yaw) * dt - s.vy * math.sin(s.yaw) * dt
        s.y = s.y + s.vx * math.sin(s.yaw) * dt + s.vy * math.cos(s.yaw) * dt
        s.yaw = s.yaw + s.omega * dt
        Ffy = -Cf * math.atan2(((s.vy + Lf * s.omega) / s.vx - delta), 1.0)
        Fry = -Cr * math.atan2((s.vy - Lr * s.omega) / s.vx, 1.0)
        s.vx = s.vx + (a - Ffy * math.sin(delta) / m + s.vy * s.omega) * dt
        s.vy = s.vy + (Fry / m + Ffy * math.cos(delta) / m - s.vx * s.omega) * dt
        s.omega = s.omega + (Ffy * Lf * math.cos(delta) - Fry * Lr) / Iz * dt
        self.state = s
        return self.state

    def update_velocity(self, v):
        self.state.vx = v
"""
m                   Mass of the vehicle [kg].
a                   Distance from front axle to COG [m].
b                   Distance from rear axle to COG [m].
Cx                  Longitudinal tire stiffness [N].
Cy                  Lateral tire stiffness [N/rad].
CA                  Air resistance coefficient [1/m].

class State2:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, vx=0.01, vy=0.0, omega=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.omega = omega

class VehicleControl(object):
    def __init__(self, x=0., y=0., velocity=0., yaw=0., wheelbase=2.65):
        self.L = wheelbase
        self.state = State2(x=x, y=y, vx=velocity, yaw=yaw)

    def get_state(self, steer, duration):
        s = self.state
        delta = steer
        dt = duration
        Lr = self.L / 2.0  # [m]
        Lf = self.L - Lr

        a = 0

        dx = vy * omega + (1 / m) * Cx * ()

        s.x = s.x + s.vx * math.cos(s.yaw) * dt - s.vy * math.sin(s.yaw) * dt
        s.y = s.y + s.vx * math.sin(s.yaw) * dt + s.vy * math.cos(s.yaw) * dt
        s.yaw = s.yaw + s.omega * dt
        Ffy = -Cf * math.atan2(((s.vy + Lf * s.omega) / s.vx - delta), 1.0)
        Fry = -Cr * math.atan2((s.vy - Lr * s.omega) / s.vx, 1.0)
        s.vx = s.vx + (a - Ffy * math.sin(delta) / m + s.vy * s.omega) * dt
        s.vy = s.vy + (Fry / m + Ffy * math.cos(delta) / m - s.vx * s.omega) * dt
        s.omega = s.omega + (Ffy * Lf * math.cos(delta) - Fry * Lr) / Iz * dt
        self.state = s
        return self.state

    def update_velocity(self, v):
        self.state.vx = v
"""