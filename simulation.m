%% Define parameters
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

%% Stage1: Escape from earth 
figure(1); clf;
fig = view(3);
daspect([1 1 1]);
grid("on")
xlim([-(R_earth+1000), (R_earth+1000)]);
ylim([-(R_earth+1000), (R_earth+1000)]);
zlim([-(R_earth+1000), (R_earth+1000)]);
theta_synodic_earth = 0.0;
T_sim = 100;

%%%%%%%% Synodic Frame %%%%%%%%%%%%
F0 = triad('Scale',8000,'LineWidth',3,'Tag','Synodic','Matrix',eye(4));

%%%%%%%% ECI Frame (Earth) %%%%%%%%
T_synodic_earth_0 = makehgtform('yrotate', deg2rad(28.58));
h_synodic_earth = hgtransform(Matrix=T_synodic_earth_0);

% Create a sphere for Earth
[earth_x, earth_y, earth_z] = sphere(20); % Create a sphere for Earth
mesh_earth = mesh(R_earth * earth_x, R_earth * earth_y, R_earth * earth_z, 'EdgeColor', 'k', ...
    'FaceAlpha',0.3,'LineWidth',0.5,'FaceColor','flat');
mesh_earth.Parent = h_synodic_earth;

% Earth frame
F_earth = triad('Scale',R_earth,'LineWidth',3,'Tag','Triad Example');
F_earth.Parent = h_synodic_earth;
text_earth = text(R_earth,0,0,'Earth', 'FontSize', 10);
text_earth.Parent = h_synodic_earth;

%%%%%%%% Space craft frame %%%%%%%%%%
x_space_craft = 0;
y_space_craft = 0;
z_space_craft = 0;

%%
for i=1:T_sim
    % Update date state of earth
    theta_synodic_earth = theta_synodic_earth + ang_vel_earth*delta_T;
    update_rotation(theta_synodic_earth, h_synodic_earth, T_synodic_earth_0, ...
        F_earth, text_earth, mesh_earth)
    
    % Update state of craft state
    x_space_craft = x_space_craft + x_dot*delta_T;
    y_space_craft = y_space_craft + y_dot*delta_T;
    y_space_craft = z_space_craft + z_dot*delta_T;
    


    pause(0.05)
    drawnow
end
%% Stage3: Landing to the moon
%%%%%%%%% MCI Frame (Moon) %%%%%%%%%%%%
% [moon_x, moon_y, moon_z] = sphere(50); % Create a sphere for Moon
% surf(distance_earth_moon + R_moon * moon_x, R_moon * moon_y, R_moon * moon_z, 'FaceColor', 'r', 'EdgeColor', 'none');
% mesh()

% Earth-Moon line
% plot3([0 distance_earth_moon], [0 0], [0 0], 'k--', 'LineWidth', 1); % Line between Earth and Moon

