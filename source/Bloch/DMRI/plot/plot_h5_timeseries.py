''' plot time series of M_INT saved in h5 format '''
import h5py
import matplotlib.pyplot as plt
import numpy as np

# fname = './results/sigbuf_0.2/b0100_ts.h5'
# fname = './results/r01_sb0.2/b0100_ts.h5'
# fname = './results/extL10_r00/b0100_ts.h5'
# fname = './results/r01/b0100_ts.h5'
fname = './results/r01_gmres/b0100_ts.h5'  # GMRES / CG signal equal
with h5py.File(fname, 'r') as h5:
    m = h5['U'].value
    t = h5['time'].value
    b = h5.attrs['b']
    h5.close()

S = np.sqrt(m[:, 0]**2+m[:, 1]**2)
plt.figure()
plt.ion()
# plt.semilogy(t, S, t, m[:, 0], t, m[:, 1], '-x')
plt.plot(t, S, t, m[:, 0], t, m[:, 1])
plt.legend((r'$|M|$', r'$M_x$', r'$M_y$'))
plt.title(r'$b = '+str(b)+'$')
plt.xlabel(r'$t$ (s)')
plt.ylabel(r'$M$')

plt.figure()
plt.semilogy(t, S)
plt.title(r'$b = '+str(b)+'$')
plt.xlabel(r'$t$ (s)')
plt.ylabel(r'$|S(t)|$')
