function [] = p1D_minimal_model_edo(s_intensity, s_duration)
%   matlab -nojvm < scriptname.m
%
%   Numerically solve Minimal ventricular model equations (single cell experiment)
%
%   FG - 2016
%   

% ======== INPUT =========

% Initial contitions
u 		= 0;
v 		= 1;
w 		= 1;
s 		= 0;
t0 		= 0;

% Total simulation time
T    		= 500;

% Time Step
dt   		= 0.1; % ms

% Discretization Method (explicit or semi-implicit)
method		= 'semi-implicit';

% Part of the ventricular hearth wall to be considered (atrial, mid, endo or epi)
part		= 'mid';

% External Stimulus
if nargin == 0
    s_intensity	= 10*1.587;   	% dimensionless
    s_duration	= dt; 		% ms
end

s_delay		= 20;           % ms

% ========================

if 	strcmp(part, 'mid')
  load './input/minimal_parameters_mid.mat'
elseif 	strcmp(part, 'endo')
  load './input/minimal_parameters_endo.mat'
elseif strcmp(part, 'epi')
  load './input/minimal_parameters_epi.mat'
else  
 load './input/minimal_parameters_atrial.mat'
end

%save minimal_parameters_atrial.mat p1_phi0 p2_phiu p3_thetav  p4_thetaw p5_thetav_minus p6_theta_0 p7_tauv1_minus p8_tauv2_minus p9_tauv_plus p10_tauw1_minus p11_tauw2_minus p12_kw_minus p13_phiw_minus p14_tau_w_plus p15_tau_fi p16_tau_o1 p17_tau_o2 p18_tau_so1 p19_tau_so2 p20_k_so p21_phi_so p22_tau_s1 p23_tau_s2 p24_ks p25_phi_s p26_tau_si p27_tauw_inf p28_w_inf 
  
P   = [t0 u v w s];

if strcmp(method, 'semi-implicit')
  % semi-implicit method
  for t = [t0 + dt:dt:T]
    eq1		= (1 - heaviside(u - p5_thetav_minus)*p7_tauv1_minus + heaviside(u - p5_thetav_minus)*p8_tauv2_minus);
    eq2		= p10_tauw1_minus + (p11_tauw2_minus - p10_tauw1_minus)*(1 + tanh(p12_kw_minus*(u - p13_phiw_minus)))/2;
    eq3		= p18_tau_so1 + (p19_tau_so2 - p18_tau_so1)*(1 + tanh(p20_k_so*(u - p21_phi_so)))/2;
    eq4		= (1 - heaviside(u - p4_thetaw))*p22_tau_s1 + heaviside(u - p4_thetaw)*p23_tau_s2;
    eq5		= (1 - heaviside(u - p6_theta_0))*p16_tau_o1 + heaviside(u - p6_theta_0)*p17_tau_o2;    
  
    v_inf	= (u < p5_thetav_minus);
    w_inf	= (1 - heaviside(u - p6_theta_0))*(1 - u/p27_tauw_inf) + heaviside(u - p6_theta_0)*p28_w_inf;
  
    % Update gating variables
    v		= (v + (dt*v_inf*(1 - heaviside(u - p3_thetav)))/eq1)/(1  +  (dt*(1 - heaviside(u - p3_thetav)))/eq1 + (dt*heaviside(u - p3_thetav))/p9_tauv_plus);
    w		= (w + (dt*w_inf*(1 - heaviside(u - p4_thetaw)))/eq2)/(1  +  (dt*(1 - heaviside(u - p4_thetaw)))/eq2 + (dt*heaviside(u - p4_thetaw))/p14_tau_w_plus);
    s		= (s + (dt*(1 + tanh(p24_ks*(u - p25_phi_s)))/2)/eq4) / (1 + dt/eq4);    

    % Current factorization
    Jfi_i	= -1/p15_tau_fi*v*heaviside(u - p3_thetav) * (p2_phiu - u + p3_thetav);
    Jfi_e	= -v*heaviside(u - p3_thetav)*p3_thetav*p2_phiu/p15_tau_fi;
    Jso_i   	= (1 - heaviside(u - p4_thetaw))/eq5;
    Jso_e	= heaviside(u - p4_thetaw)/eq3 - p1_phi0/eq5*(1 - heaviside(u - p4_thetaw));
    Jsi_e	= -heaviside(u - p4_thetaw)*w*s/p26_tau_si;
  
    % External Stimulus
    Estimulo 	= (t >= t0 + s_delay && t <= t0 + s_delay + s_duration ) * s_intensity;
    
    % Update and save solution
    u		= (u - dt*(Jfi_e + Jso_e + Jsi_e - Estimulo)) / (1.0 + dt*(Jfi_i + Jso_i));
    P	    	= vertcat(P, [t u v w s]);
  end
else
  % explicit method
  for t = [t0 + dt:dt:T]
  
    r_inf	= (phi0 < p5_thetav_minus);
    w_inf	= (1 - heaviside(phi0 - p6_theta_0))*(1 - phi0/p27_tauw_inf)...
		+ heaviside(phi0 - p6_theta_0)*p28_w_inf;
    e1 		= (1 - heaviside(phi0 - p5_thetav_minus))*p7_tauv1_minus...
		+  heaviside(phi0 - p5_thetav_minus)*p8_tauv2_minus;
    e2 		= p10_tauw1_minus...
		+ (p11_tauw2_minus - p10_tauw1_minus)*(1 + tanh(p12_kw_minus*(phi0 - p13_phiw_minus)))/2;
    e3 		= p18_tau_so1...
		+ (p19_tau_so2 - p18_tau_so1)*(1 + tanh(p20_k_so*(phi0 - p21_phi_so)))/2;
    e4		= (1 - heaviside(phi0 - p4_thetaw))*p22_tau_s1...
		+ heaviside(phi0 - p4_thetaw)*p23_tau_s2;
    e5		= (1 - heaviside(phi0 - p6_theta_0))*p16_tau_o1...
		+ heaviside(phi0 - p6_theta_0)*p17_tau_o2;
    
    r 		= r0 + dt*((1 - heaviside(phi0 - p3_thetav))*(r_inf - r0)/e1 - heaviside(phi0 - p3_thetav)*r0/p9_tauv_plus);
    w		= w0 + dt*((1 - heaviside(phi0 - p4_thetaw))*(w_inf - w0)/e2 - heaviside(phi0 - p4_thetaw)*w0/p14_tau_w_plus);
    s		= s0 + dt/e4*((1 + tanh(p24_ks*(phi0 - p25_phi_s)))/2 - s0);

    Jfi 	= -r*heaviside(phi0 - p3_thetav)*(phi0 - p3_thetav)*(p2_phiu - phi0)/p15_tau_fi;
    Jso		= (phi0 - p1_phi0)*(1.0 - heaviside(phi0 - p4_thetaw))/e5 + heaviside(phi0 - p4_thetaw)/e3;
    Jsi		= -heaviside(phi0 - p4_thetaw)*w*s/p26_tau_si;
    
    phi		= phi0 - dt*(Jfi + Jso + Jsi);
    
    phi0	= phi;
    r0      = r;
    w0      = w;
    s0      = s;
    
    P	    = vertcat(P, [t phi r w s]);
    
  end
end


% Re-scale potential to physiological values
P(:,2) = P(:,2)*85.7 - 84;

plot(P(:,1), P(:,2), 'k', 'linewidth', 2)
axis([0 500 -100 60])
grid on
ylabel('potential [mV]')
xlabel('time [ms]')