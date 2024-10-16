function step_update(delta_T)
% Update state of system 1 step forward with given timestep and thrust 
% unit vector
theta_earth = theta_earth + rot_speed_earth*delta_T;
theta_moon = theta_moon + rot_speed_moon*delta_T;
end