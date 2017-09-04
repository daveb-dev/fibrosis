from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
from solvers.ns_steady.nsestimator import NSEstimator
Axes3D
#  from matplotlib2tikz import save as tikz_save
#  from solvers.ns_steady.nssolver import NSSolver
#  from solvers.ns_steady.nsproblem import NSProblem

#  ''' optimization flags
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
#  parameters["form_compiler"]["representation"] = "quadrature"
parameters["form_compiler"]["cpp_optimize_flags"] = \
    "-O3 -ffast-math -march=native"

est = NSEstimator('input/coarc_ref.yaml', 'input/coarc_estim.yaml')
# est = NSEstimator('input/pipe_ref.yaml', 'input/pipe_estim.yaml')
# est = NSEstimator('input/pipe_ref.yaml', 'input/pipe_estim_noslip.yaml')
# est = NSEstimator('input/coarc_ref.yaml', 'input/coarc_estim_noslip.yaml')

# est.estimate(x0=20)
# x0 = np.array([1.4356, -0.1040, -1.2821, -1.1785])
# x0 = np.array([1.4, 1.0, -1.0, 2.7])
Nprm = 1
# x0 = np.ones(Nprm)*-1
x0 = np.array([-8.]*Nprm)
est.estimate(x0=x0)

est.solve_opt(est.x_opt, init=True)

err_u = norm(est.u_meas_opt.vector() -
             est.uref_meas.vector())/norm(est.uref_meas.vector())
# u1, p1 = est.pb_est.w.split(deepcopy=True)
err_dP = abs(est.pressure_drop_meas(est.p_opt) /
             est.pressure_drop_meas(est.pref) - 1.)

print('rel error U:  {0}'.format(err_u))
print('rel error dP: {0}'.format(err_dP))


def sort_fval_xi(e):
    ''' sort'''
    idx = np.argsort(est.x)
    xi = est.x[idx]
    f = est.fval[idx]
    return (xi, f)


def plot_fval_xi(elist, leg=None):
    ''' plot fval over xi for all instances of nsestimator in elist '''
    plt.figure()
    mark = itertools.cycle(['-o', '-s', '-v', '-d', '-^', '-x'])
    for e in elist:
        plt.plot(sort_fval_xi(e), mark.next())
    if leg:
        plt.legend(leg)
    plt.xlabel(r'$x_i$')
    plt.ylabel(r'Fval')
    pass


def plot_fval_multi_xi(elist):
    ''' make (x_i, x_j, fval) plots for multiparameter problems '''
    if type(elist) is list:
        nps = [e.x.shape[1] for e in elist]
        assert nps.max() == nps.min(), ('data sets need to have the same'
                                        ' number of parameters')
    else:
        elist = [elist]

    (Ni, Nj) = elist[0].x.shape
    # i = range(Ni)
    r = range(Nj)
    t = []
    for i in range(len(r) - 1):
        t += [(r[i], x) for x in r[i+1:]]

    for (i, j) in t:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        col = itertools.cycle(['k', 'b', 'c', 'r', 'm', 'g', 'y'])
        for e in elist:
            ax.scatter(e.x[:, i], e.x[:, j], e.fval, c=col.next())

        ax.set_title(r'$f(x_{i}, x_{j})$'.format(i=i, j=j))
        ax.set_xlabel(r'$x_{0}$'.format(i))
        ax.set_ylabel(r'$x_{0}$'.format(j))
        ax.set_zlabel(r'error, F')
    pass


'''
plt.ion()
plt.figure()
plt.plot(np.sqrt(est.fval)/norm(est.uref_meas.vector()))
plt.xlabel('iterations')
plt.ylabel('rel. error')

plt.figure()
plt.plot(est.x)
# plt.plot([0, x.shape[0]], [1, 1], 'k:')
plt.xlabel('iterations')
plt.ylabel(r'$x_i$')
plt.legend((r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$'))
'''


# =================================================
# gamma vs dR; Re
# dR = 0.05
#   Re = 10 --> F = 0.29358, x = [ 1.12064724,  1.05901892,  0.65927994,  0.48350214]
#       rel err U:  0.034705
#       rel err P:  0.078774
#   Re = 10 with x0i = 0.8: --> F = 0.29358, x = [ 1.12078433  1.05905093
#        0.65932685  0.48350155] --> same x for x0 = 1, 0.8 !!
#   Re = 50 --> 6.93160, x = [ 1.17760122,  0.71208976,  0.44639851,
#       -0.12745369]
#       rel err U:  0.031166
#       rel err P:  0.017210
#   Re = 50 with 2**x: very slow. F = 7.2778,
#        x = [  0.2908104 ,  -0.04384551, -1.37272634, -27.81795866]
#     2**x = [1.22332726e+00, 9.70065781e-01, 3.86160810e-01, 4.22629711e-09]
#       nit = 400, nfev = 581 (--> tols?)
#      rel error: 0.032
#   Re = 100: F = 30.195,
#             x = [ 1.24042537,  0.85812616,  0.28326602, -0.1401544]
#       rel err U: 0.031396
#       rel err P: 0.005618
#   Re = 200: F = 132.7665,
#             x = [1.24193227,  0.88653022,  0.16885018,  0.03467936]
#           rel err U:  0.0322
#           rel err dP: 0.0235
#   Re = 500:   F = 1098.792
#               x = [ 1.27600522,  0.9741858 , -0.08342871,  1.19805524]
#           rel err U:  0.03662
#           rel err dP: 0.0329
#   Re = 1000:  F = 6216.226
#               x = [ 1.43579182,  1.13800064, -0.517858  ,  4.09562415]
#           rel err U:  0.04345
#           rel err dP: 0.03268
#   Re = 1800:  F = 29302.806
#               x = [ 1.81908074,  1.20322039, -1.2902339 ,  9.91415447]
#           rel err U:  0.052366
#           rel err dP: 0.024897
#   Re = 1800:  xfun 2**x
#               F = 30819.0
#               x = [-0.07225626,  0.26410348, -4.7835653 ,  2.82082813]
#               2**x = [ 0.84332013  1.09469605  0.03868308  7.08191398]
#           rel err U:  0.0537
#           rel err dP: 0.02968
#   Re = 1800:  abs(x)  NOT CONVERGED
#               F = 30969.51
#               x = [ -1.19866007e+00,   1.26042640e+00,   2.76467261e-05,
#                   7.29467864e+00]
#           rel err U:  0.05360
#           rel err dP: 0.02442
#   Re = 1800:  x**2
#               F = 30696.30
#               x = [  1.10115820e+00,   1.12606894e+00,  -2.32011876e-06,
#                   2.70082144e+00]
#               x**2 = [  1.21269328e+00   1.26815996e+00   1.67859204e-11
#                        7.29445830e+00]
#           rel err U:  0.05360
#           rel err dP: 0.02441
#   Re = 1800:  rel norm
#               F = 0.027422
#               x = [1.81908074,  1.20322039, -1.2902339 ,  9.91415447]
#           rel err U:  0.052366
#           rel err dP: 0.024897
#   Re = 1800:  BFGS (fails horribly)
#   Re = 1800:  tikhonov 0.0025 (too low, pointless? dR**2)
#               F = 29303.02
#               x = [ 1.81905555,  1.20325488, -1.29018155,  9.91399674]
#           rel err U:  0.052366
#           rel err dP: 0.024899
#
# dR = 0.10
#   Re = 10:
#           F = 1.8918
#           x = [ 1.18727868,  1.08673361,  0.4002129 ,  0.23015756]
#       rel err U:  0.087284
#       rel err dP: 0.303896
#   Re = 50:
#           F = 49.5162
#           x = [ 1.45692102,  0.17610456, -0.24292211, -0.76506991]
#       rel err U:  0.082652
#       rel err dP: 0.09983
#   Re = 100:
#           F = 196.2377
#           x = [ 1.52822221,  0.37695293, -0.54985094, -0.93008651]
#       rel err U:  0.07945
#       rel err dP: 0.003245
#   Re = 300:
#           F = 2069.3812
#           x = [ 1.42446964,  0.68673565, -0.86093864, -0.64135203]
#       rel err U:  0.083587
#       rel err dP: 0.054972
#   Re = 500:
#           F = 6426.42
#           x = [ 1.40324824,  0.82613748, -1.04073591, -0.05981793]
#       rel err U:  0.08799
#       rel err dP: 0.06283
#   Re = 1000:
#           F = 30521.71
#           x = [ 1.65229196,  1.02494752, -1.92765777,  2.71649459]
#       rel err U:  0.09569
#       rel err dP: 0.06574
# LONG MESH:    x = [ 2.49411569,  0.67985681, -3.18173538,  5.67165108]
#               err U:  0.06126
#               err dP: 0.04839
#   Re = 1800:  ** need Nitsche beta1 = 1.e6; NM conv for 2**x method!
#           F = 1108019.92
#           x = [  1.27644172, -14.86429338,  -5.85416793,  15.48926868]
#        2**x = [  2.42240774e+00   3.35275508e-05   1.72870088e-02
#                   4.59975270e+0]
#       rel err U:  0.32
#       rel err dP: 0.8265
#

'''
# NEW: Nitsche beta 1.e6 !!
dR = [0.05]
Re = [10, 100, 400, 700, 1000, 1300, 1600]
Remax = [23, 199, 760, 1321, 1882, 2443, 2899]
errU = [0.08887, 0.13557, 0.17367, 0.19669, 0.21327, 0.22642, 0.23703]
errP = [0.30058, 0.41420, 0.44849, 0.48138, 0.51342, 0.54325, 0.56999]
# Re 100, tol 1e-3:
#     x = [   1.07797124,    2.74433055,   -1.16171313, 477.55627774]
# Re 100, tol 1e-2:
#     x = [   1.07797003,    2.7443298 ,   -1.16170782,  477.55216764]

dR.append(0.1)
Re2 = [10, 100, 400, 700, 1000, 1300]
Remax2 = [23, 199, 760, 1321, 1882, 2443]
errU2 = [0.17246, 0.26097, 0.28947, 0.30028, 0.30580, 0.31004]
errP2 = [0.78926, 0.99596, 0.90132, 0.85331, 0.82925, 0.82086]


# No-slip: dR = 0.05
Re3 = [10, 100, 300, 500, 1000, 1800]
Remax3 = [23, 199, 573, 947, 1882, 3382]
x2 = [0.35232, 3.4413, 10.2047, 16.8882, 33.3805, 59.30063629]
errU3 = [0.08810, 0.11984, 0.14774, 0.16544, 0.19363, 0.219447]
errP3 = [0.276386, 0.20564, 0.18885, 0.19228, 0.20865, 0.23506]
# dR = 0.1
x3 = [0.3512, 3.3286, 9.8891, 16.4540, 33.0052, 59.3168]
errU4 = [0.16337, 0.21611, 0.24612, 0.25946, 0.27525, 0.28504]
errP4 = [0.6798, 0.4506, 0.35998, 0.33103, 0.30888, 0.29296]

plt.figure()
plt.plot(Remax, errP, 'o-b', Remax2, errP2, 'v-b')
plt.plot(Remax3, errP3, 'o--r', Remax3, errP4, 'v--r')
plt.xlabel('Reynolds number')
plt.ylabel('rel. error pressure')
plt.legend((
    r'$\Delta R/R = 0.05$, slip', r'$\Delta R/R = 0.1$, slip',
            r'$\Delta R/R = 0.05$, no slip', r'$\Delta R/R = 0.1$, no slip'),
           loc=1)
'''

# plot this vs. noslip over max. reynolds number (REFERENCE??)

'''
Re1 = np.array([[10, 50, 100, 200, 500, 1000, 1800],
                [10, 50, 100, 300, 500, 1000, 1800]])
errU1b = np.array([[0.035, 0.0312, 0.031396, 0.033706, 0.03662, 0.04345,
                   0.052366],
                  [0.087284, 0.082652, 0.07945, 0.083587, 0.08799, 0.09569,
                   0.32]])
errP1b = np.array([[0.078774, 0.017210, 0.005618, 0.029056, 0.0329, 0.03268, 0.024897],
                  [0.303896, 0.09983, 0.003245, 0.054972, 0.06283, 0.06574,
                   0.8265]])
# 0.05
xi1 = np.array([
    [1.12064724, 1.05901892, 0.65927994, 0.48350214],
    [1.17760122, 0.71208976, 0.44639851, -0.12745369],
    [1.24042537, 0.85812616, 0.28326602, -0.1401544],
    [1.24193227, 0.88653022, 0.16885018, 0.03467936],
    [1.27600522, 0.9741858, -0.08342871, 1.19805524],
    [1.43579182, 1.13800064, -0.517858, 4.09562415],
    [1.81908074, 1.20322039, -1.2902339, 9.91415447]
])
# 0.1
xi2 = np.array([
    [1.18727868, 1.08673361, 0.4002129, 0.23015756],
    [1.45692102, 0.17610456, -0.24292211, -0.76506991],
    [1.52822221, 0.37695293, -0.54985094, -0.93008651],
    [1.42446964, 0.68673565, -0.86093864, -0.64135203],
    [1.40324824, 0.82613748, -1.04073591, -0.05981793],
    [1.65229196, 1.02494752, -1.92765777, 2.71649459],
    [1.27644172, -14.86429338, -5.85416793, 15.48926868]
])


'''
#
'''

# Re3 = [10, 100, 300, 500, 1000, 1800]
# Remax3 = [23, 199, 573, 947, 1882, 3382]
# Re1 = np.array([[10, 50, 100, 200, 500, 1000, 1800],
#                 [10, 50, 100, 300, 500, 1000, 1800]])
Re1max = np.array([[199, 573, 947, 1882, 3382],
                   [199, 573, 947, 1882, 3382]])

plt.figure()
plt.plot(Re1[0, :], errU1b[0, :], 'o-b')
plt.plot(Re1[1, :-1], errU1b[1, :-1], 'v-b')
plt.plot(Re2, errU2, 'o--r', Re2, errU3, 'v--r')
plt.xlabel('Reynolds number')
plt.ylabel('rel. error velocity')
plt.legend((r'$\Delta R/R = 0.05$, slip', r'$\Delta R/R = 0.1$, slip',
            r'$\Delta R/R = 0.05$, no slip', r'$\Delta R/R = 0.1$, no slip'),
           loc=1)

plt.figure()
plt.plot(Re1max[0, :-1], errP1b[0, 2:-1], 'o-b')
plt.plot(Re1max[1, :-1], errP1b[1, 2:-1], 'v-b')
plt.plot(Remax2[:-1], errP3[:-1], 'o--r', Remax2[:-1], errP4[:-1], 'v--r')
plt.xlabel('Reynolds number')
plt.ylabel('rel. error pressure drop')
plt.legend((r'$\Delta R/R = 0.05$, slip', r'$\Delta R/R = 0.1$, slip',
            r'$\Delta R/R = 0.05$, no slip', r'$\Delta R/R = 0.1$, no slip'),
           loc=1)

plt.figure()
plt.plot(Re1max[0, :-1], errP1b[0, 2:-1], 'o-b')
plt.plot(Re1max[1, :-1], errP1b[1, 2:-1], 'v-b')
plt.plot(Remax, errP, 'o--r', Remax2, errP2, 'v--r')
plt.xlabel('Reynolds number')
plt.ylabel('rel. error pressure drop')
plt.legend((r'$\Delta R/R = 0.05$, $\beta=1e3$', r'$\Delta R/R = 0.1$, $\beta=1e3$',
            r'$\Delta R/R = 0.05$, $\beta=1e6$', r'$\Delta R/R = 0.1$, $\beta=1e6$'),
           loc=4)

plt.figure()
plt.plot(Re1[0, :], xi1, 'o-')
plt.xlabel('Reynolds number')
plt.ylabel(r'$x_i$')
plt.legend((r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$'), loc=2)

plt.figure()
plt.plot(Re1[1, :], xi2, 'o-')
plt.xlabel('Reynolds number')
plt.ylabel(r'$x_i$')
plt.legend((r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$'), loc=2)

plt.figure()
plt.scatter(Re2, x2, c='b')
plt.scatter(Re2, x3, c='r')
plt.xlabel('Reynolds number')
plt.ylabel('estimated inlet velocity')
plt.legend((r'$\Delta R/R = 0.05$', r'$\Delta R/R = 0.1$'), loc=2)

# '''

# Nelder-Mead:
# fun: 0.036519643571320919
#        x: array([ 1.43560846, -0.10398253, -1.28205919, -1.17852873])

# pb = NSProblem('input/coarc_estim.yaml')
# pb.init()
# solver = NSSolver(pb)
# solver.solve()
