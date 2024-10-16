clc; clear
% Geometry parameters
R_earth = 6378; % radius of earth in kilometers
R_moon = 1737; % radius of moon in kilometers
distance_earth_moon = 384400; % in kilometers
rot_speed_earth = 2*pi/(24*3600); % rad/s
rot_speed_moon = 2 * pi / (27.32 * 24*3600); % rad/s
phi_synodic_earth = deg2rad(28.58); % deg 
phi_synodic_moon = deg2rad(6.68); % deg
G = 6.67430e-20; % Gravity constant in km^3*kg^-1*h^-2

% Transformation of earth, moon w.r.t synodic
T_synodic_earth_0 = makehgtform('yrotate', phi_synodic_earth);
ang_vel_earth = T_synodic_earth_0(1:3,1:3)*[0 0 1]';
T_synodic_moon_0 = makehgtform('yrotate', phi_synodic_moon);
ang_vel_moon = T_synodic_moon_0(1:3,1:3)*[0 0 1]';

% Physical parameters 
mass_earth = 5.97219e24; %kg
mass_moon = 7.34767e22; %kg
mass_sc = 49735; %kg 
mass_sc_final = 4000; %kg 

% Thrust parameters 
Ve = 3.6; % Exhaust velocity km/s
m_dot = 4000; % Mass flow rate kg/s
forceUnitVec = [0; 0; 0];

% State variable
phi_earth_space = deg2rad(60);
theta_earth_space = deg2rad(100);
time_stampt = [0];
[X_sc, V_sc] = set_initial_position(phi_earth_space, theta_earth_space, R_earth, T_synodic_earth_0);
A_sc = [0; 0; 0]; % Acceleration of space craft
mass_sc = [49735]; % Mass of space craft 
mass_fuel = [0]; % consumption fuel
theta_earth = [0]; % angle of earth
theta_moon = [0]; % angle of moon

% Simulation parameters
time_escape_eath = 0; 
time_to_zero_gra = 0;
time_land_moon = 0;
delta_T = 1; % simulation time step (hour)

% Class Space 
