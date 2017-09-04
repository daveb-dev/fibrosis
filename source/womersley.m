clear all; clc; close all
% input (CGS)
r0	= 1; 			% cylinder radius
L	= 5; 			% cylinder legnth
omega	= 1*pi;			% angular velocity of pressure gradient
visc 	= 0.035; 		% dynamic viscosity
a	= -1/L;
rho 	= 1;

%  Womersley number and static part
factScaleBessel     = besselj(0, i^(3/2)*sqrt(rho*omega/visc)*r0);

error   = 0;
error_y = 0;
j = 1;
for k = 0:2000
    time = 0.0:0.01:20;
    % import data of mesh and numerical solution
    P = csvread(strcat('numeric_sol.', num2str(k), '.csv'),1,0);

    x = P(:,4);
    y = P(:,5);
    z = P(:,6);

    solx = P(:,1);
    soly = P(:,2);
    solz = P(:,3);
    sol  = (solx.^2 + soly.^2 + solz.^2).^(.5);

    % change to polar coordinates
    r = (x.^2 + z.^2).^(0.5);

    clear i
    % analitic Womersley solution
    uy  =   real(-a/omega*(1-besselj(0,i^(3/2)*sqrt(rho*omega/visc)*r)/factScaleBessel)*exp(i*omega*time(j)));
    error_y = vertcat(error_y, norm(uy - soly, 2));
    error   = vertcat(error, norm(uy - sol, 2));    
    j = j+1;
end
error(1)   = [];
error_y(1) = [];

plot(error, '--k', 'linewidth', 2.5); hold on
figure(2)
plot(error_y, 'k', 'linewidth', 2);
grid on
xlabel('time [ms] x 10^{2}')
ylabel('error')
% legend('error', 'error y')