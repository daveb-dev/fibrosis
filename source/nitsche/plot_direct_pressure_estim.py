import matplotlib.pyplot as plt
import numpy as np

# case 1: #   d = 0,    H = 0.05
# case 2: #   d = 0,    H = 0.1
# case 3: #   d = 0.1,  H = 0.05
# case 4: #   d = 0.1,  H = 0.1

dP_ref = 4751.12907965

# no noise
PPE = [0.079, 0.107, 0.014, 0.027]
STE = [0.013, 0.031, 0.0, 0.016]
STEint = [0.008, 0.012, 0.002, 0.019]

dP = np.array([[4376.304906845848, 4689.148113036889, 4787.8868264630055],
               [4241.637866530917, 4605.707894891797, 4808.852828074914],
               [4682.796492679221, 4749.578205482842, 4763.000530788485],
               [4620.657707511375, 4826.4560706393395, 4840.775752721787]])
err_dP = np.array(
    [[0.07889159955836444, 0.01304552361587441, 0.007736634008317367],
     [0.10723581796569137, 0.030607710781558217, 0.012149480146663949],
     [0.01438238907484013, 0.0003264222332539646, 0.002498658938112186],
     [0.0274611297544044, 0.015854545252100172, 0.018868498744500473]])

PPE = dP[:, 0]
STE = dP[:, 1]
STEint = dP[:, 2]

# with noise:
# 1:
PPE_files = ['errors_PPE_d0_H0.05_noise.npz', 'errors_PPE_d0_H0.1_noise.npz',
             'errors_PPE_d0.1_H0.05_noise.npz', 'errors_PPE_noise.npz']
STE_files = ['errors_STE_d0_H0.05_noise.npz', 'errors_STE_d0_H0.1_noise.npz',
             'errors_STE_d0.1_H0.05_noise.npz', 'errors_STE_noise.npz']
STEint_files = ['errors_STEint_d0_H0.05_noise.npz',
                'errors_STEint_d0_H0.1_noise.npz',
                'errors_STEint_d0.1_H0.05_noise.npz',
                'errors_STEint_noise.npz']

PPE_files = ['errors_PPE_d0_H0.05_noise_200.npz', 'errors_PPE_d0_H0.1_noise_200.npz',
             'errors_PPE_d0.1_H0.05_noise_200.npz', 'errors_PPE_noise_200.npz']
STE_files = ['errors_STE_d0_H0.05_noise_200.npz', 'errors_STE_d0_H0.1_noise_200.npz',
             'errors_STE_d0.1_H0.05_noise_200.npz', 'errors_STE_noise_200.npz']
STEint_files = ['errors_STEint_d0_H0.05_noise_200.npz',
                'errors_STEint_d0_H0.1_noise_200.npz',
                'errors_STEint_d0.1_H0.05_noise_200.npz',
                'errors_STEint_noise_200.npz']



def plot_barplot_nonoise(PPE, STE, STEint, relative=False, dP_ref=None,
                         normalize=False):
    #  plt.figure()
    ax = plt.subplot(111)
    colors = ('b', 'k', 'c')  # , 'k', 'm'])
    labels = ('PPE', 'STE', 'STEint')
    ticklabels = (r'$\Delta R=0$, $H=0.05$', r'$\Delta R=0$, $H=0.1$',
                  r'$\Delta R=0.1$, $H=0.05$', r'$\Delta R=0.1$, $H=0.1$')

    barwidth = 0.25
    for k, (ppe, ste, stei) in enumerate(zip(PPE, STE, STEint)):
        for i, x in enumerate((ppe, ste, stei)):
            if normalize:
                x = (dP_ref - x)/dP_ref
            ax.bar(k + i*barwidth, x, barwidth, label=labels[i],
                   color=colors[i], alpha=0.4)

    ax.set_ylabel(r'relative error pressure drop')
    ax.set_title(r'Relative error in pressure drop')
    ax.set_xticks(np.arange(4) + 0.5*3*barwidth)
    ax.set_xticklabels(ticklabels)
    ax.legend(labels, loc=0)

    if relative is False and dP_ref and not normalize:
        ax.plot(ax.get_xlim(), [dP_ref]*2, 'k-', lw=2)

    #  ax.set_ylim([0, 0.5])


def plot_barplot_noise(PPE_files, STE_files, STEint_files, relative=False,
                       dP_ref=None, normalize=False):
    plt.figure()
    ax = plt.subplot(111)
    colors = ('b', 'k', 'c')  # , 'k', 'm'])
    labels = ('PPE', 'STE', 'STEint')
    ticklabels = (r'$\Delta R=0$, $H=0.05$', r'$\Delta R=0$, $H=0.1$',
                  r'$\Delta R=0.1$, $H=0.05$', r'$\Delta R=0.1$, $H=0.1$')

    barwidth = 0.25
    for k, (ppe, ste, stei) in enumerate(zip(PPE_files, STE_files,
                                             STEint_files)):
        for i, fnpz in enumerate((ppe, ste, stei)):
            data = np.load(fnpz)
            if relative:
                mean = data['mean']
                std = data['std']
            else:
                mean = data['dP'].mean()
                std = data['dP'].std()
                if normalize:
                    mean = (dP_ref - mean)/dP_ref
                    std /= dP_ref
            ax.bar(k + i*barwidth, mean, barwidth, label=labels[i],
                   color=colors[i], alpha=0.4, yerr=std, ecolor='k')

    ax.set_ylabel(r'pressure drop')
    ax.set_title(r'error in pressure drop -- NOISE')
    ax.set_xticks(np.arange(4) + 0.5*3*barwidth)
    ax.set_xticklabels(ticklabels)
    ax.legend(labels, loc=0)

    if relative is False and dP_ref and not normalize:
        ax.plot(ax.get_xlim(), [dP_ref]*2, 'k-', lw=2)
    #  ax.set_ylim([0, 0.5])


plot_barplot_noise(PPE_files, STE_files, STEint_files, dP_ref=dP_ref,
                   relative=False, normalize=False)
plot_barplot_nonoise(PPE, STE, STEint, relative=False, dP_ref=dP_ref,
                     normalize=False)


#  plt.figure()
#  plot_barplot_nonoise(PPE, STE, STEint)
#  plot_barplot_noise(PPE_files, STE_files, STEint_files)
plt.show()
