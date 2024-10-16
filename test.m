clear; clc;
m = [0; 0; 0];
for i = 2:10
    m(:,i) = m(:,i-1)+1;
end
