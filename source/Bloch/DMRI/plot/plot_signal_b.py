''' plot time series of M_INT saved in h5 format '''
import h5py
import numpy as np
import matplotlib.pyplot as plt

# brange = np.hstack((np.arange(0, 300, 50), np.arange(300, 1000, 100),
#                    np.arange(1000, 2250, 250)))

# m_TE = []
# # read all files
# for b in brange:
#     fname = 'results/b'+str(int(b)).zfill(4)+'_ts.h5'
#     h5 = h5py.File(fname, 'r')
#     m = h5['m'].value
#     # mabs = h5['m_abs'].value
#     t = h5['time'].value
#     bval = h5.attrs['b']
#     h5.close()
#     if not bval == b:
#         print('data corrupted, b attribute does not match file name!')

#     m_TE.append(np.sqrt(m[-1, 0]**2+m[-1, 1]**2))

fname = './results/sigbuf_0.0/signal_b_TE.h5'
# fname = 'results/test/signal_b_TE.h5'
h5 = h5py.File(fname, 'r')
S = h5['S'].value
b = h5['b'].value
h5.close()

Sabs = np.sqrt(S[:, 0]**2 + S[:, 1]**2)
plt.ion()
plt.figure()
plt.plot(b, Sabs, '-o')
plt.title('signal echo')
plt.xlabel(r'$b$ (s/mm^2)')
plt.ylabel(r'$S_b$')

compare_analytic = True
D = 0.00162  # +0.0006)/2
if compare_analytic:
    Sa = Sabs[0]*np.exp(-b*D)
    plt.plot(b, Sa, '-rx')

lstr = 'exact, ADC = %.2e' % D
plt.legend(('numeric', lstr))
