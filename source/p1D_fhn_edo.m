%
%   Numerically solve FitzHug-Namuno equations (single cell experiment)
%
%   FG - 2015
%

close all
clear all
clc
% ==== INPUT ==== 
c1      = 0.175;
alpha   = 0.08;
c2      = 0.03;
dt      = 0.0185; 
b       = 0.011;
d       = 0.55;
T       = 800;
t0      = 0;
phi = 0.2; r = 0; % condiciones iniciales
% ===============

P   = [t0 phi r];
for t = [t0 + dt:dt:T]
    fun = @root2d;
    x0  = [phi,r];
    aux = phi; aux2 = r;
    save phin.mat aux;   % TODO: se pueden guardar ambas variables en un solo archivo!
    save rn.mat aux2;
    x   = fsolve(fun,x0);    
    phi1 = x(1); r1 = x(2);
    P    = vertcat(P, [t phi1 r1]);
    phi  = phi1; r = r1;
end

subplot(2,1,1)
plot(P(:,1), P(:,2),'k', P(:,1), P(:,3),'--k')  %,'color',[j/(j+1) j/(10) j^2/(j^3+0.0001)])
hold on; grid on
xlabel('time')
legend('phi','r')

subplot(2,1,2)
plot(P(:,2), P(:,3),'k')  %, 'color',[j/(j+1) j/(10) j^2/(j^3+0.0001)])
hold on
grid on
xlabel('phi')
ylabel('r')

% === IMPLICIT METHOD ===

