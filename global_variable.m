clc; clear
R_earth = 6378; % radius of earth in kilometers
R_moon = 1737; % radius of moon in kilometers
distance_earth_moon = 384400; % in kilometers
ang_vel_earth = 2*pi/24; % rad/h
ang_vel_moon = 2 * pi / (27.32 * 24); % rad/h
T_escape_eath = 0; 
T_to_zero_gra = 0;
T_land_moon = 0;
delta_T = 1; % simulation time step (hour)