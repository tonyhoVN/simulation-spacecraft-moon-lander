from math import sin, cos, tan, atan2, pi, radians, degrees
import numpy as np
from scipy.spatial.transform import Rotation as R
import transformations.transformations as TF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a helper function for plotting axis triads
def plot_triad(ax, origin, scale=1, linewidth=2, tag=''):
    """Plots a 3D triad (X, Y, Z axes) at a given origin."""
    # X axis
    ax.quiver(origin[0], origin[1], origin[2], scale, 0, 0, color='r', linewidth=linewidth)
    # Y axis
    ax.quiver(origin[0], origin[1], origin[2], 0, scale, 0, color='g', linewidth=linewidth)
    # Z axis
    ax.quiver(origin[0], origin[1], origin[2], 0, 0, scale, color='b', linewidth=linewidth)
    
    # Label for the frame
    ax.text(origin[0] + scale, origin[1], origin[2], f'{tag} X', color='r', fontsize=10)
    ax.text(origin[0], origin[1] + scale, origin[2], f'{tag} Y', color='g', fontsize=10)
    ax.text(origin[0], origin[1], origin[2] + scale, f'{tag} Z', color='b', fontsize=10)

def set_initial_position(phi, theta, radius, R_synodic_earth_0):
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    # Position
    position = np.dot(
        R_synodic_earth_0, np.array([x, y, z])
    ) 
    velocity = np.dot(
        R_synodic_earth_0, np.array([0, 0, 0])
    )
    thrust_unit_force = np.dot(R_synodic_earth_0, np.array([0, 0, 1]))
    return position, velocity, thrust_unit_force


class SpaceCraft:
    def __init__(
        self,
        mass=49735,
        Ve=3.6,
        m_dot=4000,
        X_init=np.zeros(3),
        V_init=np.zeros(3),
        A_init=np.zeros(3),
        thrust_unit=np.zeros(3),
    ):
        self.x = X_init
        self.v = V_init
        self.a = A_init
        self.thrust_unit = thrust_unit
        self.m = [mass]
        self.Ve = Ve
        self.m_dot = m_dot

    def step_update(self, force = np.zeros(3), thrust_unit = np.zeros(3), time_step = 0):        # mass update
        m = self.m[-1] + self.m_dot*time_step
        a = (force + self.m_dot*self.v[-1,:])/self.m[-1]
        v = self.v[-1,:] + self.a[-1,:]*time_step
        x = self.x[-1,:] + self.v[-1,:]*time_step

        # append 
        self.m.append(m)
        self.a = a
        self.v = np.append(self.v, v, axis=0)
        self.x = np.append(self.x, x, axis=0)

    def visualize(self, ax, scale=1, linewidth=2, tag=''):
        # coordinate axis
        x_current = self.x[-1,:]
        plot_triad(ax, x_current, scale, linewidth, tag)
        # scatter 
        ax.scatter(x_current[0], x_current[1], x_current[2], color='b')

        

