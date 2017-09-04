from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
from solvers.ns_steady.nssolver import NSSolver
from solvers.ns_steady.nsproblem import NSProblem
from solvers.ns_steady.nsestimator import NSEstimator

est = NSEstimator('input/coarc_valid_estim.yaml',
                  'input/coarc_valid_estim.yaml')

x0 = 4.
est.estimate(x0=x0, opt_target=0)

err_u = norm(est.u_meas.vector() -
             est.uref_meas.vector())/norm(est.uref_meas.vector())
u1, p1 = est.pb_est.w.split(deepcopy=True)
err_dP = abs(est.pressure_drop_meas(p1)/est.pressure_drop_meas(est.pref) - 1.)

print('rel error U:  {0}'.format(err_u))
print('rel error dP: {0}'.format(err_dP))
plt.ion()
plt.figure()
plt.plot(np.sqrt(est.fval)/norm(est.uref_meas.vector()))
plt.xlabel('iterations')
plt.ylabel('rel. error')

x = np.array(est.x)
plt.figure()
plt.plot(x)
# plt.plot([0, x.shape[0]], [1, 1], 'k:')
plt.xlabel('iterations')
plt.ylabel(r'$x_i$')
plt.legend((r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$'))
