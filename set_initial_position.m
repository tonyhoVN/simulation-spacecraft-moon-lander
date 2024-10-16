function [X_space, V_space] = set_initial_position(phi, theta, R_earth, T0)
    x0 = R_earth*cos(phi)*cos(theta);
    y0 = R_earth*cos(phi)*sin(theta);
    z0 = R_earth*sin(phi);
    X_space = T0(1:3,1:3)*[x0; y0; z0];
    w_0 = T0(1:3,1:3)*[0 0 1]';
    V_space = cross(w_0, X_space);
end