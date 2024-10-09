import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

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

# Define parameters
R_earth = 6378  # radius of Earth in kilometers
R_moon = 1737   # radius of Moon in kilometers
distance_earth_moon = 384400  # in kilometers
ang_vel_earth = 2 * np.pi / 24  # rad/h
ang_vel_moon = 2 * np.pi / (27.32 * 24)  # rad/h
T_escape_earth = 0
T_to_zero_gra = 0
T_land_moon = 0
delta_T = 1  # simulation time step (hour)

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plt.ion()

# Set aspect ratio and limits
ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-(R_earth+1000), (R_earth+1000)])
ax.set_ylim([-(R_earth+1000), (R_earth+1000)])
ax.set_zlim([-(R_earth+1000), (R_earth+1000)])
ax.grid(True)

# Stage 1: Escape from Earth
theta_synodic_earth = 0.0
T_sim = 100

# Create a sphere for Earth
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
earth_x = R_earth * np.cos(u) * np.sin(v)
earth_y = R_earth * np.sin(u) * np.sin(v)
earth_z = R_earth * np.cos(v)

# Plot Earth
ax.plot_surface(earth_x, earth_y, earth_z, color='b', alpha=0.3, edgecolor='k')

# Plot the Synodic Frame (for visualization purposes)
plot_triad(ax, origin=[0, 0, 0], scale=8000, linewidth=0.5, tag='Synodic')

# ECI Frame (Earth) – Triad centered at Earth
T_synodic_earth_0 = np.eye(3)  # Placeholder for rotation matrix
plot_triad(ax, origin=[0, 0, 0], scale=R_earth, linewidth=0.5, tag='Earth')

# Label for Earth
ax.text(R_earth, 0, 0, 'Earth', fontsize=12)



# Display the plot
plt.title('3D View of Earth and Frames')
plt.show()

