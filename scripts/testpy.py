from utils import *
import transformations.transformations as TF
from scipy.spatial.transform import Rotation as R

file_path = os.path.join(os.getcwd(),"..","data")
save_file_path = os.path.join(file_path,"scraft.csv")

# Geometry parameters
r_earth = 6378; # radius of earth in kilometers
r_moon = 1737; # radius of moon in kilometers
distance_earth_moon = 384400; # in kilometers
rot_speed_earth = 0 # 2*pi/(24*3600); # rad/s
rot_speed_moon = 0 # 2 * pi / (27.32 * 24 * 3600); # rad/s
inclination = radians(5.14)
phi_synodic_earth = radians(28.58); # deg 
phi_synodic_moon = radians(6.68); # deg
G = 6.67430e-20; # Gravity constant in km^3*kg^-1*h^-2
g_zero_point = np.array([346019.66, 0, 0])

# # Transformation of earth, moon w.r.t synodic
R_synodic_earth_0 = R.from_euler('y', phi_synodic_earth, degrees=False).as_matrix()
ang_vel_earth = np.dot(R_synodic_earth_0, np.array([0, 0, 1]))*rot_speed_earth
R_synodic_moon_0 = R.from_euler('y', phi_synodic_moon, degrees=False).as_matrix()
# ang_vel_moon = np.dot(R_synodic_moon_0, np.array([0, 0, 1]))*rot_speed_moon
ang_vel_moon = np.zeros(3)

# Physical parameters
mass_earth = 5.97219e24  # kg
mass_moon = 7.34767e22  # kg
mass_sc = 49735  # kg (initial mass of spacecraft)
mass_sc_final = 4000  # kg (final mass of spacecraft after fuel consumption)

# Thrust parameters
Ve = 12  # Exhaust velocity in km/s
m_dot = 4000  # Mass flow rate in kg/s

# State variables
phi_earth_space = np.deg2rad(60)
theta_earth_space = np.deg2rad(100)
time_stamp = [0]
theta_synodic_space = np.deg2rad(139.0933)
fire_angle = np.deg2rad(42.2394)

X_sc, V_sc, A_sc, thrust_unit = set_initial_new(
    theta = theta_synodic_space,
    alpha = fire_angle,
    r_earth=r_earth
)

# breakpoint()

# Simulation parameters
time_escape_eath = 0.0 
time_to_zero_gra = 0.0
time_land_moon = 0.0
delta_T = 1; # simulation time step (hour)

# Class Space 
Earth = Planet(
    mass=mass_earth,
    radius=r_earth,
    ang_vel=ang_vel_earth,
    theta=0.0,
    phi=phi_synodic_earth,
    distance_synodic=0,
    incline_synodic=0
)

Moon = Planet(
    mass=mass_moon,
    radius=r_moon,
    ang_vel=ang_vel_moon,
    theta=0.0,
    phi=phi_synodic_moon,
    distance_synodic=distance_earth_moon,
    incline_synodic=inclination
)

space_craft = SpaceCraft(
    save_file=save_file_path,
    mass=mass_sc,
    mass_final=mass_sc_final,
    Ve = Ve,
    m_dot=m_dot,
    X_init=X_sc,
    V_init=np.zeros(3),
    A_init=np.zeros(3),
    thrust_unit=thrust_unit
)

space_craft.reset(save_file=save_file_path, m=mass_sc, x=X_sc, v=V_sc, a=A_sc)

time_step = 1
steps = int(1*3600)

# Burning process
time_phase1 = 6


############### STAGE 1 ################
for i in range(0, 1428, 1):
    if i < 6:
        thrust_unit_input = thrust_unit 
    else:
        thrust_unit_input = None

    space_craft.step_update(G=G, earth=Earth, moon=Moon, thrust_unit=thrust_unit_input, time_step=time_step)

# print(space_craft.x)
# print(space_craft.v)
# print(space_craft.a)
# print(space_craft.m)
# Visualize
visualize_system(
    spacecraft_state_file=save_file_path,
    spacecraft=space_craft,
    earth=Earth,
    moon=Moon,
    g_zero_point=g_zero_point,
    show_earth=True,
    show_moon=False,
    scale_axis=1
)

############### STAGE 2 ################
count = 0
steps = int(70*24*3600)
previous = False
for i in range(0, steps, 10):
    # Break at the back of Moon
    if space_craft.x[0] > Moon.x[0] and abs(space_craft.x[1]) < 50:
        break
    
    if space_craft.x[0] < 0 and abs(space_craft.x[1]) < 50:
        if not previous: 
            count += 1
            if count == 1:
                thrust_unit_input = np.array([0,-1,0])
                space_craft.step_update(G=G, earth=Earth, moon=Moon, thrust_unit=thrust_unit_input, time_step=1)
                space_craft.step_update(G=G, earth=Earth, moon=Moon, thrust_unit=thrust_unit_input, time_step=1)
                # space_craft.step_update(G=G, earth=Earth, moon=Moon, thrust_unit=thrust_unit_input, time_step=1)
                thrust_unit_input = None
            elif count == 2:
                thrust_unit_input = np.array([0,-0.11,0])
                space_craft.step_update(G=G, earth=Earth, moon=Moon, thrust_unit=thrust_unit_input, m_dot=4000, time_step=1)
                space_craft.step_update(G=G, earth=Earth, moon=Moon, thrust_unit=thrust_unit_input, m_dot=4000, time_step=1)
                space_craft.step_update(G=G, earth=Earth, moon=Moon, thrust_unit=thrust_unit_input, m_dot=4000, time_step=1)
                thrust_unit_input = None
            else:
                thrust_unit_input = None
            previous = True
        else: 
            thrust_unit_input = None
    else:
        thrust_unit_input = None
        previous = False

    # Update state
    space_craft.step_update(G=G, earth=Earth, moon=Moon, thrust_unit=thrust_unit_input, time_step=time_step)



# Visualize
visualize_system(
    spacecraft_state_file=save_file_path,
    spacecraft=space_craft,
    earth=Earth,
    moon=Moon,
    g_zero_point=g_zero_point,
    show_earth=True,
    show_moon=True,
    scale_axis=1
)
# '''


######## PHASE 3 ##########
steps = int(50*24*3600)
for i in range(0, steps, 10):
    if i==0:
        thrust_unit_input = np.array([-1,0,0])
        space_craft.step_update(G=G, earth=Earth, moon=Moon, thrust_unit=thrust_unit_input, Ve=2.19, m_dot=1000, time_step=1)   
        thrust_unit_input = None
    else:
        thrust_unit_input = None

    # Update state
    space_craft.step_update(G=G, earth=Earth, moon=Moon, thrust_unit=thrust_unit_input, time_step=time_step)

visualize_system(
    spacecraft_state_file=save_file_path,
    spacecraft=space_craft,
    earth=Earth,
    moon=Moon,
    g_zero_point=g_zero_point,
    show_earth=True,
    show_moon=True,
    scale_axis=1
)   