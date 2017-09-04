close all
clear all
clc

ERROR = importdata('./monodomain/ex4/error_L2')
%ERROR = importdata('./data40.csv')
%ERROR = ERROR.data(1:end - 1, 1:3)
%ERROR(:,2) = []


plot(ERROR(:,1), ERROR(:,2), 'k', 'linewidth', 2)
xlabel('Time [ms]', 'FontSize', 14)
ylabel('Error-L2', 'FontSize', 14)
grid on
axis([0 200 0 0.15])

%xlabel('Radial Coordinate [cm]')
%ylabel('Temperature [C]')