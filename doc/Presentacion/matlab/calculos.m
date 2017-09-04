clear all; clc; close all;

L   = 25;    % Longitud del cauce [km]
L_g = 18.2;  % Longitud desde el centroide de la cuenca hasta el punto de control [km]
S   = 0.192; % Pendiente media de la cuenca
tc  = 1.75;  % Tiempo de retención

tp = 0.323*(L*L_g/sqrt(S))^0.422;  % [hr]
tB = 5.377*tp^0.805;               % [hr]
qp = 144.141*tp^(-0.796);          % [lt/s/mm/km^2]
tu = tp/5.5;                       % [hr]


%% Diagrama unitario sintético tipo Linsley

razont = [0.00 0.30 0.50 0.60 0.75 1.00 1.30 1.50 1.80 2.30 2.70 tB/tp];  % Diagrama unitario adimensional (según DGA)
razonq = [0.00 0.20 0.40 0.60 0.80 1.00 0.80 0.60 0.40 0.20 0.10 0
    ];  % Diagrama unitario adimensional (según DGA)

t0 = razont*tp;
t1 = t0 + 0.88*ones(size(t0));
q = razonq*qp;

plot(t0,q,'.',t1,q,'.','markersize',20)
axis([0 25 0 40])

intq = trapz(t0,q)*3600;
qc   = q*1000000/intq