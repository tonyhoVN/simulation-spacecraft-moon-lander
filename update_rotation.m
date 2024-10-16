function [] = update_rotation(theta, h, T0, Frame, text, mesh)

T_synodic_earth = makehgtform('zrotate', theta);
set(h,'Matrix', T0*T_synodic_earth);
h_mesh = hgtransform(Matrix= T0*makehgtform('zrotate', -theta));
set(Frame,'Parent',h);
set(text, 'Parent', h);
set(mesh,'Parent',h_mesh);

end