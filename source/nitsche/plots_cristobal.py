import numpy as np
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

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

# NOSLIP VS NAVIERSLIP BETA 1e6
plt.figure()
plt.plot(Remax[2:], errP[2:], 'o-b', lw=2, ms=10)
plt.plot(Remax2[2:], errP2[2:], 'v-b', lw=2, ms=10)
plt.plot(Remax3[2:], errP3[2:], 'o--r', lw=2, ms=10)
plt.plot(Remax3[2:], errP4[2:], 'v--r', lw=2, ms=10)
plt.xlabel('Reynolds number')
plt.ylabel('rel. error pressure')
plt.legend((
    r'$\Delta R/R = 0.05$, Navier-slip', r'$\Delta R/R = 0.1$, Navier-slip',
    r'$\Delta R/R = 0.05$, no slip', r'$\Delta R/R = 0.1$, no slip'),
    loc=1)
tikz_save('tmp/errP_noslip_slip_beta1e6.tikz', figurewidth='\\figurewidth',
          figureheight='\\figureheight')

# NOSLIP ONLY
plt.figure()
# plt.plot(Remax[2:], errP[2:], 'o-b', lw=2, ms=10)
# plt.plot(Remax2[2:], errP2[2:], 'v-b', lw=2, ms=10)
plt.plot(Remax3[2:], errP3[2:], 'o--r', lw=2, ms=10)
plt.plot(Remax3[2:], errP4[2:], 'v--r', lw=2, ms=10)
plt.xlabel('Reynolds number')
plt.ylabel('rel. error pressure')
plt.legend((
    # r'$\Delta R/R = 0.05$, Navier-slip', r'$\Delta R/R = 0.1$, Navier-slip',
    r'$\Delta R/R = 0.05$, no slip', r'$\Delta R/R = 0.1$, no slip'),
    loc=1)
plt.ylim([0.1, 0.5])
tikz_save('tmp/errP_noslip.tikz', figurewidth='\\figurewidth',
          figureheight='\\figureheight')
# '''

# plot this vs. noslip over max. reynolds number (REFERENCE??)

# '''
Re1 = np.array([[10, 50, 100, 300, 500, 1000, 1800],
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
Re1max = np.array([[199, 573, 947, 1882, 3382],
                   [199, 573, 947, 1882, 3382]])

# VELOCITY ERROR
plt.figure()
plt.plot(Re1[0, :], errU1b[0, :], 'o-b')
plt.plot(Re1[1, :-1], errU1b[1, :-1], 'v-b')
plt.plot(Re2, errU2, 'o--r', Re2, errU3, 'v--r')
plt.xlabel('Reynolds number')
plt.ylabel('rel. error velocity')
plt.legend((r'$\Delta R/R = 0.05$, slip', r'$\Delta R/R = 0.1$, slip',
            r'$\Delta R/R = 0.05$, no slip', r'$\Delta R/R = 0.1$, no slip'),
           loc=1)


# PRESSURE NOSLIP VS NAVIERSLIP BETA 1e3
# TODO *** INSERT PRESSURE RE=300 INTO errP1b
plt.figure()
plt.plot(Re1max[0, 1:-1], errP1b[0, 3:-1], 'o-b', lw=2, ms=10)
plt.plot(Re1max[1, 1:-1], errP1b[1, 3:-1], 'v-b', lw=2, ms=10)
plt.plot(Remax3[2:-1], errP3[2:-1], 'o--r', lw=2, ms=10)
plt.plot(Remax3[2:-1], errP4[2:-1], 'v--r', lw=2, ms=10)
plt.xlabel('Reynolds number')
plt.ylabel('rel. error pressure drop')
plt.legend((r'$\Delta R/R = 0.05$, Navier-slip', r'$\Delta R/R = 0.1$, Navier-slip',
            r'$\Delta R/R = 0.05$, no slip', r'$\Delta R/R = 0.1$, no slip'),
           loc=1)
plt.ylim([0, 0.5])

tikz_save('tmp/errP_noslip_slip_beta1e3.tikz', figurewidth='\\figurewidth',
          figureheight='\\figureheight')

# PRESSURE NAVIERSLIP ONLY, NITSCHE BETA 1e6 vs 1e3
plt.figure()
plt.plot(Re1max[0, 1:-1], errP1b[0, 3:-1], 'o-b', lw=2, ms=10)
plt.plot(Re1max[1, 1:-1], errP1b[1, 3:-1], 'v-b', lw=2, ms=10)
plt.plot(Remax[2:-2], errP[2:-2], 'o--r', lw=2, ms=10)
plt.plot(Remax2[2:-1], errP2[2:-1], 'v--r', lw=2, ms=10)
plt.xlabel('Reynolds number')
plt.ylabel('rel. error pressure drop')
plt.legend((r'$\Delta R/R = 0.05$, $\beta=1e3$', r'$\Delta R/R = 0.1$, $\beta=1e3$',
            r'$\Delta R/R = 0.05$, $\beta=1e6$', r'$\Delta R/R = 0.1$, $\beta=1e6$'),
           loc=2)
plt.xlim([200, 2000])

tikz_save('tmp/errP_slip_beta1e3_1e6.tikz', figurewidth='\\figurewidth',
          figureheight='\\figureheight')

# PRESSURE NAVIERSLIP ONLY, NITSCHE BETA 1e6 vs 1e3
plt.figure()
plt.plot(Re1max[0, 1:-1], errU1b[0, 3:-1], 'o-b', lw=2, ms=10)
plt.plot(Re1max[1, 1:-1], errU1b[1, 3:-1], 'v-b', lw=2, ms=10)
plt.plot(Remax[2:-2], errU[2:-2], 'o--r', lw=2, ms=10)
plt.plot(Remax2[2:-1], errU2[2:-1], 'v--r', lw=2, ms=10)
plt.xlabel('Reynolds number')
plt.ylabel('rel. error velocity')
plt.legend((r'$\Delta R/R = 0.05$, $\beta=1e3$', r'$\Delta R/R = 0.1$, $\beta=1e3$',
            r'$\Delta R/R = 0.05$, $\beta=1e6$', r'$\Delta R/R = 0.1$, $\beta=1e6$'),
           loc=2)
plt.xlim([200, 2000])

# plt.figure()
# plt.plot(Re1[0, :], xi1, 'o-')
# plt.xlabel('Reynolds number')
# plt.ylabel(r'$x_i$')
# plt.legend((r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$'), loc=2)

# plt.figure()
# plt.plot(Re1[1, :], xi2, 'o-')
# plt.xlabel('Reynolds number')
# plt.ylabel(r'$x_i$')
# plt.legend((r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$'), loc=2)

# plt.figure()
# plt.scatter(Re2, x2, c='b')
# plt.scatter(Re2, x3, c='r')
# plt.xlabel('Reynolds number')
# plt.ylabel('estimated inlet velocity')
# plt.legend((r'$\Delta R/R = 0.05$', r'$\Delta R/R = 0.1$'), loc=2)
