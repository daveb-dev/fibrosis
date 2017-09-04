import numpy as np
import matplotlib.pyplot as plt

alpha = 0.08 
c1    = 0.175
c2    = .03
b     = 0.011
d     = .55

dt   = 1
T    = 800; time = np.arange(0, T, dt)


phi0 = 0.2
r0   = 0
r, phi = np.zeros(int(T/dt)), np.zeros(int(T/dt))

t = dt; i = 0;
while t <= T:
    r1 = (b*phi0 + r0/dt)/(1/dt +b*d)

    # Semi-implicit #1 (strong)
#    phi1 = (phi0/dt - c2*r1)/(1/dt - c1*phi0 + c1*phi0*phi0 +c1*alpha - c1*alpha*phi0)
    # Semi-implicit #2 (weak)    
    phi1 = (c1*phi0*phi0 - c1*phi0*phi0*phi0 + c1*alpha*phi0*phi0 - c2*r1 + phi0/dt)/(1/dt + c1*alpha)

    r[i] = r1;
    phi[i] = phi1; 
    i = i + 1
    
    r0, phi0 = r1, phi1
    
    t = t + dt

fig1 = plt.figure()
plt.plot(time, phi, '-r', label = 'normalized potential')
plt.plot(time, r, '-k', label = 'gating variable')
plt.legend()    
plt.xlabel('time [ms]')
plt.grid('on')
plt.show()
