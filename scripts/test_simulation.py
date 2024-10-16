
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Create a rotation matrix for a given angle and axis
def rotation_matrix(axis, theta):
    """
    Returns the rotation matrix for a given axis and angle theta (in radians).
    Axis must be a 3-element numpy array.
    """
    axis = axis / np.sqrt(np.dot(axis, axis))  # Normalize the axis
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                     [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])

# Plotting function to draw the frames
def plot_frame(ax, origin, R, label, color):
    """ Draws a 3D frame (coordinate axes) with a rotation matrix R """
    # X-axis
    ax.quiver(*origin, *R[:,0], color=color[0], label=f'{label} x-axis', length=1)
    # Y-axis
    ax.quiver(*origin, *R[:,1], color=color[1], label=f'{label} y-axis', length=1)
    # Z-axis
    ax.quiver(*origin, *R[:,2], color=color[2], label=f'{label} z-axis', length=1)

# Initialize figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the fixed frame (Earth-centered frame)
origin_fixed = np.array([0, 0, 0])
I_fixed = np.eye(3)  # Identity matrix for fixed frame

# Define the rotating frame (initially aligned with the fixed frame)
I_rotating = np.eye(3)  # Starts aligned with the fixed frame

# Set up limits for the axes
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])

# Axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Animation function to rotate the frame around a given axis
def update(frame):
    ax.cla()  # Clear the plot
    
    # Plot fixed frame (does not rotate)
    plot_frame(ax, origin_fixed, I_fixed, 'Fixed', ['r', 'g', 'b'])

    # Rotate the rotating frame by a small angle (theta increases with frames)
    theta = np.radians(frame)  # Convert degrees to radians
    R = rotation_matrix(np.array([0, 0, 1]), theta)  # Rotate around the Z-axis
    
    # Apply the rotation to the rotating frame
    I_rot = np.dot(R, I_rotating)
    
    # Plot the rotating frame
    plot_frame(ax, origin_fixed, I_rot, 'Rotating', ['c', 'm', 'y'])
    
    # Update view for better visualization
    # ax.view_init(elev=30., azim=frame * 2)  # Rotate the view slightly as well

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=100)

plt.legend()
plt.show()