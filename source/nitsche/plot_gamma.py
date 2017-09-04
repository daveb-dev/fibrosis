import numpy as np
import matplotlib.pyplot as plt

R = 1.
f0 = 0.2
Xi = 1.0
Xo = 2.0
L = 2.0
x0 = Xi + L/2.0

dR = 0.1
mu = 0.01


def f(x):
    if x > Xi and x < Xi + L:
        return R*(1 - f0/2.*(1 + np.cos(2*np.pi*(x-x0)/L)))
    else:
        return R

x = np.linspace(0, Xi + Xo + L, 100)
y = np.zeros(len(x))
for i, xi in enumerate(x):
    y[i] = f(xi)

g = 1 - 2*y*mu/(2*y*dR + dR**2)
# makes no sense, Poiseuille sol for thinner tube almost equal..
g = 1 + 2*y*mu/(y**2 - (R + dR)**2)


plt.ion()
plt.figure()
plt.plot(x, y, x, g)
plt.ylim((0, 1.1))
