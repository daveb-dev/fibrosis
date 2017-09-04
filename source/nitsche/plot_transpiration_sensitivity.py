import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
Axes3D  # just to supress flake8 warning
#  from matplotlib2tikz import save as tikz_save

path = './experiments/transpiration_sensitivity/'


def plot_error_bars_UP(grouplist, save='', labels='', ticklabels='',
                       plot_STEi=False):
    ''' plot errorbars for U, dP for all data sets.

    Args:
        grouplist       list of (list of) optimization cases, e.g.,
            s1_f0.6_noise
            grouplist[i][:] are a group of the same type and plotted at
                different ticks with the same color
        save        save plot: None, png or tikz
        plot_STEi (bool)  include STEint error in plot
    '''
    assert type(grouplist) is list
    if type(grouplist[0]) is not list:
        grouplist = [grouplist]

    # number of ticks
    num_ticks = len(grouplist[0])
    # number of bars per atick
    num_bars_u = len(grouplist)
    num_bars_p = len(grouplist)
    index = np.arange(num_ticks)

    if not labels or not (type(labels) is list and len(labels) == num_bars_u):
        labels = ['group' + str(i) for i in range(num_bars_u)]
    if not ticklabels:
        ticklabels = index

    elif plot_STEi:
        num_bars_p += 1

    # bar plot options
    barwidth_p = 1./(num_bars_p + 1)
    barwidth_u = 1./(num_bars_u + 1)

    plt.figure()    # velocity error
    ax1 = plt.subplot(111)
    plt.figure()    # pressure error
    ax2 = plt.subplot(111)
    err_dP_stei = []
    colors = itertools.cycle(['b', 'r', 'c', 'k', 'm'])
    for ct, (group, label) in enumerate(zip(grouplist, labels)):
        col = next(colors)
        err_u, err_dP = [], []
        for dset in group:
            with open(path + 'results/' + dset + '/estim_results.dat', 'rb') \
                    as fin:
                data = pickle.load(fin, encoding='latin1')
                # encoding='latin1' for python2 compatibility
                err_u.append(data['err_u'])
                err_dP.append(data['err_dP'])
                if plot_STEi and len(err_dP_stei) < num_ticks:
                    err_dP_stei.append(data['err_dP_stei'])

        ax1.bar(index + ct*barwidth_u, err_u, barwidth_u, label=label,
                color=col, alpha=0.4)
        ax2.bar(index + ct*barwidth_p, err_dP, barwidth_p, label=label,
                color=col, alpha=0.4)

    if plot_STEi:
        ax2.bar(index + (ct + 1)*barwidth_p, err_dP_stei, barwidth_p,
                label='STEint', color=next(colors), alpha=0.4)

    ax1.set_xlabel(r'dataset')
    ax1.set_ylabel(r'$\mathcal{E}[U]$')
    ax1.set_title(r'$L_2$ error of $\mathcal{I}_H^1(u)$ wrt measurement, ' +
                  r'$\tilde u\in \mathcal{P}_H^1$')
    ax1.set_xticks(index + 0.5*num_bars_u*barwidth_u)
    ax1.set_xticklabels(ticklabels)
    ax1.legend(loc=0)

    ax1.set_xlabel(r'dataset')
    ax2.set_ylabel(r'$\mathcal{E}[\Delta P]$')
    ax2.set_title(r'relative error of $\Delta P$ wrt reference solution')
    ax2.set_xticks(index + 0.5*num_bars_p*barwidth_p)
    ax2.set_xticklabels(ticklabels)
    ax2.legend(loc=0)

    pass


def plot_compare_navslip_noslip_steint(grouplist, save='', labels='',
                                       ticklabels=''):
    ''' Bar plot for number of function evaluations in BFGS run.

    Args:
        grouplist   list of type [navierslip_1, navierslip_i], [noslip_1, ...].
                    STEint is taken from the last data set (overwritten every
                    time for simplicity)
    '''

    # number of ticks
    num_ticks = len(grouplist[0])
    # number of bars per atick
    num_bars = len(grouplist)
    index = np.arange(num_ticks)

    if not labels:
        labels = ['group' + str(i) for i in range(num_bars)]
    if not ticklabels or not (type(ticklabels) is list and len(ticklabels) ==
                              num_ticks):
        ticklabels = index

    # bar plot options
    barwidth = 1./(num_bars + 1)
    barwidth_p = 1./(num_bars + 2)

    plt.figure()    # velocity error
    ax1 = plt.subplot(111)
    plt.figure()    # velocity error
    ax2 = plt.subplot(111)
    colors = itertools.cycle(['b', 'k', 'c', 'r', 'm'])
    err_dP_stei = []
    for ct, (group, label) in enumerate(zip(grouplist, labels)):
        col = next(colors)
        err_u, err_dP = [], []
        err_dP_stei.append([])
        for dset in group:
            with open(path + 'results/' + dset + '/estim_results.dat', 'rb') \
                    as fin:
                data = pickle.load(fin, encoding='latin1')
                # encoding='latin1' for python2 compatibility
                err_u.append(data['err_u'])
                err_dP.append(data['err_dP'])
                err_dP_stei[ct].append(data['err_dP_stei'])

        ax1.bar(index + ct*barwidth, err_u, barwidth, label=label,
                color=col, alpha=0.6)
        ax2.bar(index + ct*barwidth_p, err_dP, barwidth_p, label=label,
                color=col, alpha=0.6)
    ax2.bar(index + (ct + 1)*barwidth_p, err_dP_stei[-1], barwidth_p,
            label=labels[-1], color='c', alpha=0.6)

    # check err_dP_stei variance
    check_stei = np.array(err_dP_stei).std(axis=0)
    print('STD STEint error between data groups:   {0}'.format(check_stei))
    assert np.allclose(check_stei, 0), 'STEint error differs'

    ax1.set_xlabel(r'dataset')
    ax1.set_ylabel(r'$\mathcal{E}[U]$')
    ax1.set_title(r'Velocity measurement error')
    ax1.set_xticks(index + 0.5*num_bars*barwidth)
    ax1.set_xticklabels(ticklabels)
    ax1.legend(loc=0)

    ax1.set_xlabel(r'dataset')
    ax2.set_ylabel(r'$\mathcal{E}[\Delta P]$')
    ax2.set_title(r'relative error in $\Delta P$')
    ax2.set_xticks(index + 0.5*(num_bars + 1)*barwidth_p)
    ax2.set_xticklabels(ticklabels)
    ax2.legend(loc=0)

    pass


def barplot_compare_statistics(ylst, yerrlst, labels='', ticklabels='',
                               ylabel='', title=''):
    ''' Plot pressure drop/v (rel err) barchart with errorbars
    '''
    if type(ylst) is not list:
        ylst = [ylst]
        yerrlst = [yerrlst]

    num_ticks = len(ylst)
    num_bars = len(ylst[0])

    if not ticklabels:
        ticklabels = range(num_ticks)

    # bar plot options
    barwidth = 1./(num_bars + 1)

    plt.figure()
    ax1 = plt.subplot(111)
    colors = ['b', 'k', 'c', 'r', 'm']

    print(ylst)
    print(yerrlst)
    for i, (yset, yerrset) in enumerate(zip(ylst, yerrlst)):
        for k, (y, yerr) in enumerate(zip(yset, yerrset)):
            ax1.bar(i + k*barwidth, y, barwidth, color=colors[k], alpha=0.4,
                    yerr=yerr, ecolor='k')

    #  ax1.set_xlabel(r'dataset')
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.set_xticks(np.arange(num_ticks) + 0.5*num_bars*barwidth)
    ax1.set_xticklabels(ticklabels)
    if labels:
        ax1.legend(labels, loc=0)
    pass


def plot_bars_feval(grouplist, save='', labels='', ticklabels=''):
    ''' Bar plot for number of function evaluations in BFGS run.

    Args:
        grouplist       list of (list of) optimization cases, e.g.,
            s1_f0.6_noise
            grouplist[i][:] are a group of the same type and plotted at
                different ticks with the same color
        save        save plot: None, png or tikz
    '''
    assert type(grouplist) is list
    if type(grouplist[0]) is not list:
        grouplist = [grouplist]

    # number of ticks
    num_ticks = len(grouplist[0])
    # number of bars per atick
    num_bars = len(grouplist)
    index = np.arange(num_ticks)

    if not labels or not (type(labels) is list and len(labels) == num_bars):
        labels = ['group' + str(i) for i in range(num_bars)]
    if not ticklabels or not (type(ticklabels) is list and len(ticklabels) ==
                              num_ticks):
        ticklabels = index

    # bar plot options
    barwidth = 1./(num_bars + 1)

    plt.figure()    # velocity error
    ax1 = plt.subplot(111)
    colors = itertools.cycle(['b', 'r', 'c', 'k', 'm'])
    for ct, (group, label) in enumerate(zip(grouplist, labels)):
        col = next(colors)
        feval = []
        for dset in group:
            with open(path + 'results/' + dset + '/estim_results.dat', 'rb') \
                    as fin:
                data = pickle.load(fin, encoding='latin1')
                # encoding='latin1' for python2 compatibility
                feval.append(len(data['f']))

        ax1.bar(index + ct*barwidth, feval, barwidth, label=label,
                color=col, alpha=0.4)

    ax1.set_xlabel(r'dataset')
    ax1.set_ylabel(r'num feval')
    ax1.set_title(r'Number of function evaluations during optimization')
    ax1.set_xticks(index + 0.5*num_bars*barwidth)
    ax1.set_xticklabels(ticklabels)
    ax1.legend(loc=0)
    pass


def plot_scatter3d_fval_over_x(datasets, save='', labels=''):
    ''' Plot the minimum of the objective function over the parameters
        x = (beta, gamma).
    '''
    if type(datasets) is not list:
        datasets = [datasets]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = itertools.cycle(['k', 'b', 'c', 'r', 'm', 'g', 'y'])
    for dset in datasets:
        fpath = path + 'results/' + dset + '/estim_results.dat'
        with open(fpath, 'rb') as fin:
            data = pickle.load(fin, encoding='latin1')
            # encoding='latin1' for python2 compatibility
            print(data['x'].shape)
            if data['x'].shape[1] < 4:
                x2 = 2**data['x'][:, 2]   # beta
                x1 = np.ones(x2.shape)*0.   # gamma
            else:
                x1 = 2**data['x'][:, 2]   # gamma
                x2 = 2**data['x'][:, 3]   # beta
            fval = data['f']
        col = next(colors)
        ax.scatter(x1, x2, fval, c=col, alpha=1.0, s=40)
        ax.plot3D(x1, x2, fval, '-'+str(col))

    ax.set_title(r'Objective function $F=F(\gamma, \beta)$')
    ax.set_xlabel(r'$x_1 = \gamma$')
    ax.set_ylabel(r'$x_2 = \beta$')
    ax.set_zlabel(r'$F$')
    if labels:
        ax.legend(labels)
    pass


def plot_lines3d_fval_over_x(f0lst, beta=None, gamma=None, normalize=True,
                             prefix='s2', suffix='', save='', labels=''):
    ''' Plot the minimum of the objective function over the parameters
        x = (beta, gamma).

    Args:
        datasets    list of datasets of type 's2_f0.4_beta|gamma1000'
    '''
    if beta and not gamma:
        x2lst = beta
        label2 = r'$\beta$'
        label1 = r'$\gamma$'
        parameter = 'beta'
    elif gamma and not beta:
        x2lst = gamma
        label1 = r'$beta$'
        label2 = r'$\gamma$'
        parameter = 'gamma'
    else:
        raise Exception('specify beta OR gamma')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = itertools.cycle(['k', 'b', 'c', 'r', 'm', 'g', 'y'][:len(f0lst)])

    for x2, f0 in itertools.product(x2lst, f0lst):
        col = next(colors)
        dset = [prefix, 'f{:1.1f}'.format(f0), parameter + str(x2)]
        if suffix:
            dset.append(suffix)
        dset = '_'.join(dset)
        fpath = (path + 'results/' + prefix + '/f{}/'.format(f0) + dset +
                 '/estim_results.dat')

        with open(fpath, 'rb') as fin:
            data = pickle.load(fin, encoding='latin1')
            # encoding='latin1' for python2 compatibility
        if data['x'].shape[1] > 1:
            raise Exception('optimizing more than just 1 paramenter '
                            '(beta|gamma). Makes no sense here!')
        x1 = 2**data['x']   # gamma|beta
        fval = data['f']
        if normalize:
            fval /= fval.max()

        x2arr = np.ones(x1.shape)*x2
        ax.scatter(x1, x2arr, fval, c=col, alpha=1.0, s=40)
        ax.plot3D(x1, x2arr, fval, '-'+str(col))

    legend = [r'$f_0 = {}$'.format(f0_) for f0_ in f0lst]

    title = r'Objective function $F=F(\gamma, \beta)$'
    if normalize:
        title += ', normalized'
    ax.set_title(title)
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.set_zlabel(r'$F$')
    ax.legend(legend, loc=0)
    pass


def plot_bruteforce_3d(f0lst, stride=5, normalize=True, prefix='s2', suffix='',
                       save='', labels='', smooth=False):
    ''' Plot the minimum of the objective function over the parameters
        x = (beta, gamma).

    Args:
        datasets    list of datasets of type 's2_f0.4_beta|gamma1000'
    '''

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    colors = itertools.cycle(['k', 'b', 'c', 'r', 'm', 'g', 'y'][:len(f0lst)])

    handles1 = []
    handles2 = []
    handles3 = []
    fmax = []
    for f0 in f0lst:
        verts1 = []
        verts2 = []
        col = next(colors)
        dset = [prefix, 'f{f:1.1f}_bruteforce']
        if suffix:
            dset.append(suffix)
        dset = '_'.join(dset)
        fpath = (path + 'results/' + prefix + '/f{f}/' + dset +
                 '/estim_results.dat').format(f=f0)

        with open(fpath, 'rb') as fin:
            data = pickle.load(fin, encoding='latin1')
            # encoding='latin1' for python2 compatibility
        assert data['x'].shape[1] == 2

        x1 = data['x'][:, 0]   # gamma
        x2 = data['x'][:, 1]   # beta
        fval = data['f']
        num_a = len(np.where(x1 == x1[0])[0])
        num = len(np.where(x1 == x1[-1])[0])
        if not num_a == num:
            if num_a == num + 1:
                # eliminate first row!
                x1 = x1[1:]
                x2 = x2[1:]
                fval = fval[1:]
            else:
                raise Exception('paramters do not match')
        if normalize:
            fval /= fval.max()
        fmax.append(fval.max())

        legstr = r'$f_0 = {}$'.format(f0)
        # fig1: f(beta), gamma fixed
        x1_ = [x1[i*stride*num:(i*stride+1)*num] for i in
               range(int(num/stride))]
        x2_ = [x2[i*stride*num:(i*stride+1)*num] for i in
               range(int(num/stride))]
        fval_ = [fval[i*stride*num:(i*stride+1)*num] for i in
                 range(int(num/stride))]
        #  verts1 = []
        for t1, t2, fv in zip(x1_, x2_, fval_):
            if smooth:
                spl = scipy.interpolate.UnivariateSpline(t2, fv, s=0)
                xn1 = np.linspace(t1.min(), t1.max(), 100)
                xn2 = np.linspace(t2.min(), t2.max(), 100)
                fvn = spl(xn2)
                h, = ax1.plot3D(xn1, xn2, fvn, '-'+str(col), label=legstr,
                                lw=2)
                ax1.scatter(t1, t2, fv, c=col, alpha=.8, s=40)
            else:
                h, = ax1.plot3D(t1, t2, fv, '-'+str(col), label=legstr, lw=2)
                ax1.scatter(t1, t2, fv, c=col, alpha=.8, s=40)
            #  for t in t1:
            #      t2_ = np.hstack((t2[0], t2, t2[-1]))
            #      fv_ = np.hstack((0, fv, 0))
            #      verts1.append(list(zip(t2_, fv_)))
        handles1.append(h)
        for t1 in x1_:
            verts1.append([[np.array(x2_).min(), 0.0],
                           [np.array(x2_).max(), 0.0],
                           [np.array(x2_).max(), 1.1],
                           [np.array(x2_).min(), 1.1]])
        zs1 = np.unique(x1_)

        # could also sort the data for x2 and use same procedure as below
        x1_ = [x1[i::num] for i in range(0, num, stride)]
        x2_ = [x2[i::num] for i in range(0, num, stride)]
        fval_ = [fval[i::num] for i in range(0, num, stride)]
        for t1, t2, fv in zip(x1_, x2_, fval_):
            if smooth:
                spl = scipy.interpolate.UnivariateSpline(t1, fv, s=0)
                xn1 = np.linspace(t1.min(), t1.max(), 100)
                xn2 = np.linspace(t2.min(), t2.max(), 100)
                fvn = spl(xn1)
                h, = ax2.plot3D(xn1, xn2, fvn, '-'+str(col), label=legstr,
                                lw=2)
                ax2.scatter(t1, t2, fv, c=col, alpha=.8, s=40)
            else:
                h, = ax2.plot3D(t1, t2, fv, '-'+str(col), label=legstr, lw=2)
                ax2.scatter(t1, t2, fv, c=col, alpha=.8, s=40)
        handles2.append(h)

        for tmp in x2_:
            verts2.append([[np.array(x1_).min(), 0.0],
                           [np.array(x1_).max(), 0.0],
                           [np.array(x1_).max(), 1.1],
                           [np.array(x1_).min(), 1.1]])
        zs2 = np.unique(x2_)

        handles3.append(ax3.scatter(x1, x2, fval, c=col, alpha=.6, s=40,
                                    label=legstr))

    #  legend = [r'$f_0 = {}$'.format(f0_) for f0_ in f0lst]

    title = r'Objective function $F=F(\gamma, \beta)$'
    if normalize:
        title += ', normalized'

    if not normalize:
        for v in verts1:
            v[-2:][1][1] = max(fmax)*1.05
            v[-2:][0][1] = max(fmax)*1.05
        for v in verts2:
            v[-2:][1][1] = max(fmax)*1.05
            v[-2:][0][1] = max(fmax)*1.05
        # do
    poly = PolyCollection(verts1, facecolors=[cc('k')])
    ax1.add_collection3d(poly, zs=zs1, zdir='x')

    poly = PolyCollection(verts2, facecolors=[cc('k')])
    ax2.add_collection3d(poly, zs=zs2, zdir='y')

    for ax, h in zip((ax1, ax2, ax3), (handles1, handles2, handles3)):
        ax.set_title(title)
        ax.set_xlabel(r'$x_1 = \gamma(x)$')
        ax.set_ylabel(r'$x_2 = \beta$')
        ax.set_zlabel(r'$F$')
        ax.legend(handles=h, loc=0)
    pass


def cc(arg):
    return colorConverter.to_rgba(arg, alpha=0.1)


if __name__ == '__main__':
    def barplots_f0():
        grouplist = [
            ['s1_f0.2', 's1_f0.4', 's1_f0.6'],
            ['s1_f0.2_noslip', 's1_f0.4_noslip', 's1_f0.6_noslip'],
        ]
        labels = ['Navier-slip/transpiration', 'no-slip']
        ticklabels = [r'$f_0 = 0.2$', r'$f_0 = 0.4$', r'$f_0 = 0.6$']
        plot_error_bars_UP(grouplist, labels=labels, ticklabels=ticklabels,
                           plot_STEi=False)

        grouplist = [
            ['s1_f0.2_noise', 's1_f0.4_noise', 's1_f0.6_noise'],
            ['s1_f0.2_noslip_noise', 's1_f0.4_noslip_noise',
             's1_f0.6_noslip_noise'],
        ]
        labels = ['nslip/transp w/ noise', 'no-slip w/ noise']
        ticklabels = [r'$f_0 = 0.2$', r'$f_0 = 0.4$', r'$f_0 = 0.6$']
        plot_error_bars_UP(grouplist, labels=labels, ticklabels=ticklabels,
                           plot_STEi=True)

    def barplots_noise():
        grouplist = [
            ['s1_f0.6_noise', 's1_f0.6_noslip_noise'],
        ]
        for i in range(2, 11):
            grouplist.append(
                ['s1_f0.6_noise_{0}'.format(i),
                 's1_f0.6_noslip_noise_{0}'.format(i)]
            )
        labels = ['noise {0}'.format(i) for i in range(len(grouplist))]
        ticklabels = [r'nav/trans', r'noslip']
        plot_error_bars_UP(grouplist, labels=labels, ticklabels=ticklabels,
                           plot_STEi=True)

    def barplots_compare_slip_ns_stei():
        grouplist = [
            ['s1_f0.6', 's1_f0.6_noise_seed10', 's1_f0.6_noise_seed11'],
            ['s1_f0.6_noslip', 's1_f0.6_noslip_noise_seed10',
             's1_f0.6_noslip_noise_seed11'],
        ]
        labels = ['Navslip', 'No-slip', 'STEint']
        ticklabels = ['no noise', 'noise seed 10', 'noise seed 11']
        plot_compare_navslip_noslip_steint(grouplist, labels=labels,
                                           ticklabels=ticklabels)

    def barplots_compare_slip_ns_stei_statistics():
        grouplist = [
            ['s1_f0.6', 's1_f0.6_noslip']
        ]
        grouplist = [
            ['s1_f0.6_noise_seed{s}', 's1_f0.6_noslip_noise_seed{s}']
        ]
        labels = ['Navslip', 'No-slip', 'PPE', 'STE', 'STEint']
        plot_compare_navslip_noslip_steint(grouplist, labels=labels,
                                           seeds=None,
                                           direct=('PPE', 'STE', 'STEint'))

    def barplots_gm():
        grouplist = [
            ['s1_f0.6', 's1_f0.6_gm0', 's1_f0.6_gm-const'],
        ]
        labels = ['no noise']
        ticklabels = [r'$\gamma = \gamma(x)$', r'$\gamma=0$',
                      r'$\gamma=const$']
        plot_error_bars_UP(grouplist, labels=labels, ticklabels=ticklabels,
                           plot_STEi=False)
        plot_bars_feval(grouplist, labels=labels, ticklabels=ticklabels)

        grouplist = [
            ['s1_f0.6_noise', 's1_f0.6_gm0_noise', 's1_f0.6_gm-const_noise'],
        ]
        for i in range(2, 11):
            grouplist.append(
                ['s1_f0.6_noise_{0}'.format(i),
                 's1_f0.6_gm0_noise_{0}'.format(i),
                 's1_f0.6_gm-const_noise_{0}'.format(i)]
            )
        labels = ['noise {0}'.format(i) for i in range(len(grouplist))]
        ticklabels = [r'$\gamma = \gamma(x)$', r'$\gamma=0$',
                      r'$\gamma=const$']
        plot_error_bars_UP(grouplist, labels=labels, ticklabels=ticklabels,
                           plot_STEi=False)

        plot_bars_feval(grouplist, labels=labels, ticklabels=ticklabels)

    def scatter_3d_f0():
        dsets = ['s1_f0.2', 's1_f0.4', 's1_f0.6']
        labels = [r'$f_0 = 0.2$', r'$f_0 = 0.4$', r'$f_0 = 0.6$']
        plot_scatter3d_fval_over_x(dsets, labels=labels)
        plt.title(r'no noise, $\gamma=\gamma(x)$, $\Delta R=0.1$')

        dsets = ['s1_f0.2_noise', 's1_f0.4_noise', 's1_f0.6_noise']
        labels = [r'$f_0 = 0.2$', r'$f_0 = 0.4$', r'$f_0 = 0.6$']
        plot_scatter3d_fval_over_x(dsets, labels=labels)
        plt.title(r'10% noise, $\gamma=\gamma(x)$, $\Delta R=0.1$')

        dsets = ['s1_f0.6_noise', 's1_f0.6_noise_2', 's1_f0.6_noise_3']
        labels = [r'0.6, noise #1', r'0.6, noise #2', r'0.6, noise #3']
        plot_scatter3d_fval_over_x(dsets, labels=labels)
        plt.title(r'$f_0=0.6$, 3 realizations of 10% noise, '
                  r'$\gamma=\gamma(x)$, $\Delta R=0.1$')

    def scatter_3d_gm():
        dsets = ['s1_f0.6_gm-const', 's1_f0.6_gm0', 's1_f0.6']
        labels = [r'$\gamma = const$', r'$\gamma=0$', r'$\gamma=\gamma(x)$']
        plot_scatter3d_fval_over_x(dsets, labels=labels)
        plt.title('no noise, f=0.6')

        dsets = ['s1_f0.6_gm-const_noise_seed10', 's1_f0.6_gm0_noise_seed10',
                 's1_f0.6_noise_seed10']
        labels = [r'$\gamma = const$', r'$\gamma=0$', r'$\gamma=\gamma(x)$']
        plot_scatter3d_fval_over_x(dsets, labels=labels)
        plt.title('noise seed10, f=0.6')

        dsets = ['s1_f0.6_gm-const_noise_seed11', 's1_f0.6_gm0_noise_seed11',
                 's1_f0.6_noise_seed11']
        labels = [r'$\gamma = const$', r'$\gamma=0$', r'$\gamma=\gamma(x)$']
        plot_scatter3d_fval_over_x(dsets, labels=labels)
        plt.title('noise seed11, f=0.6')

    def lines_3d_beta_gamma():
        f0 = [0.0, 0.2, 0.4, 0.6]
        beta = [0, 500, 1000, 1500, 2000]
        plot_lines3d_fval_over_x(f0, beta=beta)

        gamma = [0.0, 0.25, 0.5, 0.75, 1.0]
        plot_lines3d_fval_over_x(f0, gamma=gamma)
        pass

    def lines_3d_bruteforce():
        f0 = [0.0, 0.2, 0.4, 0.6]
        plot_bruteforce_3d(f0, stride=2, smooth=True, normalize=False,
                           prefix='s2')
        pass

    def barplots_noise_statistics():
        npz = [path + '/results/s1_f0.6_noise_seed0-99_statistics.npz',
               path + '/results/s1_f0.6_noslip_noise_seed0-99_statistics.npz']
        dat = [path + '/results/s1_f0.6/estim_results.dat',
               path + '/results/s1_f0.6_noslip/estim_results.dat']
        ulst1 = []
        uerrlst1 = []
        dPlst1 = []
        dPerrlst1 = []
        for f in npz:
            data = np.load(f)
            ulst1.append(data['err_u'])
            uerrlst1.append(data['err_u_std'])
            dPlst1.append(data['err_dP'])
            dPerrlst1.append(data['err_dP_std'])
        dPlst1.append(data['err_dP_ppe'])
        dPlst1.append(data['err_dP_ste'])
        dPlst1.append(data['err_dP_stei'])
        dPerrlst1.append(data['err_dP_ppe_std'])
        dPerrlst1.append(data['err_dP_ste_std'])
        dPerrlst1.append(data['err_dP_stei_std'])

        ulst1 = np.array(ulst1)
        uerrlst1 = np.array(uerrlst1)
        dPlst1 = np.array(dPlst1)
        dPerrlst1 = np.array(dPerrlst1)

        ulst2 = []
        uerrlst2 = []
        dPlst2 = []
        dPerrlst2 = []
        for f in dat:
            with open(f, 'rb') as fin:
                data = pickle.load(fin, encoding='latin1')
            ulst2.append(data['err_u'])
            uerrlst2.append(0)
            dPlst2.append(data['err_dP'])
            dPerrlst2.append(0)
        dPlst2.append(data['err_dP_ppe'])
        dPlst2.append(data['err_dP_ste'])
        dPlst2.append(data['err_dP_stei'])
        dPerrlst2.append(0)
        dPerrlst2.append(0)
        dPerrlst2.append(0)

        ulst2 = np.array(ulst2)
        uerrlst2 = np.array(uerrlst2)
        dPlst2 = np.array(dPlst2)
        dPerrlst2 = np.array(dPerrlst2)

        ticklabels = ['noise avg', 'no noise']
        ulabels = ['Navier-slip', 'No-slip']
        dPlabels = ['Navier-slip', 'No-slip', 'PPE', 'STE', 'STEint']

        barplot_compare_statistics([ulst1, ulst2], [uerrlst1, uerrlst2],
                                   labels=ulabels, ticklabels=ticklabels,
                                   title='Velocity measurement error',
                                   ylabel=r'$\mathcal{E}[U]$')

        barplot_compare_statistics([dPlst1, dPlst2], [dPerrlst1, dPerrlst2],
                                   labels=dPlabels, ticklabels=ticklabels,
                                   title='Pressure drop error',
                                   ylabel=r'$\mathcal{E}[\Delta P]$')


    #  barplots_gm()
    #  scatter_3d_f0()
    #  scatter_3d_gm()
    #  barplots_f0()
    #  barplots_noise()
    #  barplots_compare_slip_ns_stei()

    #  lines_3d_beta_gamma()

    plt.close('all')
    barplots_noise_statistics()
    #  lines_3d_bruteforce()

    plt.show()
