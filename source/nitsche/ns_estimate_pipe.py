from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib2tikz import save as tikz_save
# from solvers.ns_steady.nssolver import NSSolver
# from solvers.ns_steady.nsproblem import NSProblem
from solvers.ns_steady.nsestimator import NSEstimator

est = NSEstimator('input/pipe_ref.yaml', 'input/pipe_estim.yaml')
est2 = NSEstimator('input/pipe_ref.yaml', 'input/pipe_estim.yaml')
# est = NSEstimator('input/pipe_ref.yaml', 'input/pipe_estim_noslip.yaml')
est.options['estimation']['method'] = 'Nelder-Mead'
est2.options['estimation']['method'] = 'bruteforce'

Nprm = 1
# x0 = np.ones(Nprm)*2.5
# x0 = 0.067857
# x0 = 0.6  # for prefactor
x0 = [-0.5]*Nprm  # for 2**x
est.estimate(x0=x0)
est2.estimate(x0=x0)

idx = np.argsort(est.x)
x1 = est.x[idx]
f1 = est.fval[idx]

idx = np.argsort(est2.x)
x2 = est2.x[idx]
f2 = est2.fval[idx]

plt.ion()
plt.figure()
plt.plot(x1, f1, 'o-', x2, f2, '-s')
plt.xlabel(r'Parameter $\delta R = 2^\theta$')
plt.ylabel('fval')
plt.title('Fval(x) convexity')
plt.legend(('bruteforce', 'Nelder-Mead'), loc=0)

err_u = norm(est.u_meas.vector() -
             est.uref_meas.vector())/norm(est.uref_meas.vector())
u1, p1 = est.pb_est.w.split(deepcopy=True)
err_dP = abs(est.pressure_drop_meas(p1)/est.pressure_drop_meas(est.pref) - 1.)

print('rel error U:  {0}'.format(err_u))
print('rel error dP: {0}'.format(err_dP))
plt.figure()
plt.plot(np.sqrt(est.fval)/norm(est.uref_meas.vector()))
plt.plot(np.sqrt(est2.fval)/norm(est2.uref_meas.vector()))
plt.xlabel('iterations')
plt.ylabel('rel. error')
plt.title('Fval over iterations')
plt.legend(('bruteforce', 'Nelder-Mead'), loc=0)

plt.figure()
plt.plot(x1)
plt.plot(x2)
# plt.plot([0, x.shape[0]], [1, 1], 'k:')
plt.xlabel('iterations')
plt.ylabel(r'$x_i$')
plt.title('Parameters over iterations')
plt.legend(('bruteforce', 'Nelder-Mead'), loc=0)
# plt.legend((r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$'))
