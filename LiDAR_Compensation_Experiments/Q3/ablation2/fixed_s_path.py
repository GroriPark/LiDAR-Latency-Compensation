"""Fixed arc-length geometry, with a separately integrated speed schedule."""
import numpy as np


def tracking_velocity(heading, speed, curvature, position_error, actual_velocity,
                      gain=1.5, response_time=3.0, damping=2.0):
    """Curvature feedforward plus lateral position/velocity feedback in world XY.

    response_time approximates the inner velocity loop lag, not a LiDAR model.
    Keep the requested speed magnitude; heading remains the path tangent.
    """
    tangent = np.array([np.cos(heading), np.sin(heading)])
    normal = np.array([-tangent[1], tangent[0]])
    forward = max(0., float(np.dot(actual_velocity, tangent)))
    lateral = (response_time*forward**2*curvature
               + gain*np.dot(position_error, normal)
               - damping*np.dot(actual_velocity, normal))
    velocity = speed*tangent+lateral*normal
    return velocity*(speed/max(np.linalg.norm(velocity), 1e-12))


class FixedSPath:
    def __init__(self, speed_min=5., speed_max=10.):
        self.length = 160.0
        self.s = np.linspace(0, self.length, 16001)
        u = self.s / self.length
        # Two fixed 24 m turns, each changing heading by 70 degrees.
        self.heading = np.full_like(u, np.deg2rad(-35.))
        self.curvature = np.zeros_like(u)
        for a, sign in ((.22, 1), (.62, -1)):
            q = np.clip((u-a)/.15, 0, 1)
            self.heading += sign*np.deg2rad(70)*(q-np.sin(2*np.pi*q)/(2*np.pi))
            self.curvature += sign*np.deg2rad(70)/(self.length*.15)*(1-np.cos(2*np.pi*q))
        ds = np.diff(self.s)
        tangent = np.column_stack((np.cos(self.heading), np.sin(self.heading)))
        self.xy = np.vstack((np.zeros(2), np.cumsum((tangent[:-1]+tangent[1:])*ds[:, None]/2, axis=0)))
        knots = [0, .10, .22, .37, .49, .62, .77, .87, 1]
        levels = [0, 1, 0, 0, 1, 0, 0, 1, 0]
        self.speed = np.empty_like(u)
        for a, b, lo, hi in zip(knots[:-1], knots[1:], levels[:-1], levels[1:]):
            mask = (u >= a) & (u <= b)
            q = (u[mask]-a)/(b-a)
            self.speed[mask] = speed_min+(speed_max-speed_min)*(lo+(hi-lo)*(3*q*q-2*q*q*q))
        self.time = np.r_[0., np.cumsum(ds*(1/self.speed[:-1]+1/self.speed[1:])/2)]
        self.duration = float(self.time[-1])

    def at(self, elapsed):
        s = float(np.interp(elapsed, self.time, self.s))
        return self.at_distance(s)

    def project(self, xy):
        """Nearest segment projection; geometry is fixed and has no self crossings."""
        delta = np.diff(self.xy, axis=0)
        fraction = np.clip(np.sum((np.asarray(xy)-self.xy[:-1])*delta, axis=1)
                           / np.sum(delta*delta, axis=1), 0, 1)
        projected = self.xy[:-1]+fraction[:, None]*delta
        index = int(np.argmin(np.sum((projected-xy)**2, axis=1)))
        return float(self.s[index]+fraction[index]*(self.s[index+1]-self.s[index]))

    def at_distance(self, s):
        s = float(np.clip(s, 0, self.length))
        xy = np.array([np.interp(s, self.s, self.xy[:, i]) for i in range(2)])
        heading = float(np.interp(s, self.s, self.heading))
        speed = float(np.interp(s, self.s, self.speed))
        curvature = float(np.interp(s, self.s, self.curvature))
        return s/self.length, xy, heading, speed, speed*curvature
