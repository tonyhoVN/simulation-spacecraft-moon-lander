import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants from the provided image
earth_radius = 6378  # in km
moon_radius = 1738   # in km
earth_moon_distance = 38440  # in km
earth_axial_tilt = 23.44  # in degrees
moon_axial_tilt = 6.68    # in degrees
moon_orbit_inclination = 5.14  # in degrees

# Simulation settings
def simulate_earth_moon_rotation(simulation_time=24, time_step=0.1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Axes for Earth and Moon
    earth_axes = np.array([[0, 0, earth_radius * 2], [0, 0, -earth_radius * 2]])
    moon_axes = np.array([[0, 0, moon_radius * 2], [0, 0, -moon_radius * 2]])

    # Convert angles to radians
    earth_axial_tilt_rad = np.radians(earth_axial_tilt)
    moon_axial_tilt_rad = np.radians(moon_axial_tilt)
    
    def rotate_z(angle):
        """ Create a 3D rotation matrix around the Z axis """
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

    # Time variables
    time = 0
    earth_rotation_angle = 0
    moon_rotation_angle = 0

    while time < simulation_time:
        ax.cla()  # Clear the plot
        
        # Update Earth rotation
        earth_rotation_angle += time_step * (2 * np.pi / 24)  # Earth rotates 360° in 24 hours
        
        # Update Moon rotation
        moon_rotation_angle += time_step * (2 * np.pi / (27.32 * 24))  # Moon rotation takes 27.32 days
        
        # Rotate Earth axis
        earth_rot_matrix = rotate_z(earth_rotation_angle)
        rotated_earth_axes = np.dot(earth_rot_matrix, earth_axes.T).T
        
        # Rotate Moon axis
        moon_rot_matrix = rotate_z(moon_rotation_angle)
        rotated_moon_axes = np.dot(moon_rot_matrix, moon_axes.T).T
        
        # Plot Earth
        earth_sphere = plt.Circle((0, 0), earth_radius, color='blue', alpha=0.5)
        ax.plot([0, 0], [0, 0], [-earth_radius, earth_radius], 'r')  # Earth axis
        ax.quiver(0, 0, 0, earth_radius, earth_radius * np.tan(earth_axial_tilt_rad), 0, color='r', label="Earth Axis")

        # Plot Moon
        moon_pos = [earth_moon_distance, 0, 0]  # Simple 2D position for moon
        ax.scatter(*moon_pos, color='gray', s=moon_radius / 10)  # Simplified for 3D

        # Plot Moon axis
        ax.quiver(*moon_pos, moon_radius * np.cos(moon_axial_tilt_rad), moon_radius * np.sin(moon_axial_tilt_rad), moon_radius, color='green', label="Moon Axis")

        # Set labels and axis limits
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        # ax.set_zlabel('Z (km)')
        # ax.set_xlim([-500000, 500000])
        # ax.set_ylim([-500000, 500000])
        # ax.set_zlim([-500000, 500000])
        ax.set_aspect('auto')

        plt.pause(0.001)  # Update the plot

        time += time_step

    plt.show()

# Run the simulation
simulate_earth_moon_rotation(simulation_time=48, time_step=0.1)
