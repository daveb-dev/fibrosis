from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import random
import string
import warnings
import copy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib2tikz import save as tikz_save
from solvers.ns_steady.nssolver import NSSolver
from solvers.ns_steady.nsproblem import NSProblem
from solvers.ns_steady.nsestimator import NSEstimator

#  ''' optimization flags
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
#  parameters["form_compiler"]["representation"] = "quadrature"
parameters["form_compiler"]["cpp_optimize_flags"] = \
    "-O3 -ffast-math -march=native"

#  '''
path = './experiments/navierslip_transpiration_test/'
plt.ion()


def plot_fopt_over_Re(prefs, Res, betas, methods, Re_scale=1., STEint=False):
    ''' Plot F_opt(x) over the Reynolds number for each beta and optimization
    method. '''
    if type(prefs) is not list:
        prefs = [prefs]
    if type(methods) is not list:
        methods = [methods]
    if type(Res) not in (list, np.array):
        Res = [Res]
    if type(betas) not in (list, np.array):
        betas = [betas]

    Xo = []
    Fo = []
    Eu = []
    Edp = []
    ifig = fignum()
    fig1 = plt.figure(ifig)
    fig2 = plt.figure(ifig + 1)
    # fig3 = plt.figure(ifig + 2)
    # fig4 = plt.figure(ifig + 3)
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    # ax3 = fig3.add_subplot(111)
    # ax4 = fig4.add_subplot(111)
    ax1.set_title('Error in velocity')
    ax2.set_title('Error in pressure drop')
    # ax3.set_title('minimum of objective function')
    # ax4.set_title('optimal parameter')

    if len(prefs) == 2:
        col = itertools.cycle(['r', 'b', 'r', 'b'])  # 'b', 'r', 'm', 'g'])
        mark = itertools.cycle(['d', 'd', 'o', 'd'])  # , 's', '^', 'd', 'x'])
        marksize = itertools.cycle([10, 10])  # , 's', '^', 'd', 'x'])
        linst = itertools.cycle(['-', '-', '-', '-'])
    elif len(prefs) == 4:
        col = itertools.cycle(['r', 'r', 'b', 'b'])  # 'b', 'r', 'm', 'g'])
        mark = itertools.cycle(['o', 'd', 'o', 'd'])  # , 's', '^', 'd', 'x'])
        marksize = itertools.cycle([10, 10])  # , 's', '^', 'd', 'x'])
        linst = itertools.cycle(['--', '-', '--', '-'])
    else:
        col = itertools.cycle(['k', 'c', 'b', 'r', 'm', 'g'])
        mark = itertools.cycle(['o', 'v', 's', '^', 'd', 'x'])
        marksize = itertools.cycle([10, 10])  # , 's', '^', 'd', 'x'])
        linst = itertools.cycle(['-'])

    for (pref, meth, beta) in itertools.product(prefs, methods, betas):
        betastr = 'b' + str(int(beta))
        x_opt = []
        f_opt = []
        err_u = []
        err_dp = []
        for Re in Res:
            Restr = 'Re' + str(int(Re))
            fname = '_'.join([pref, meth, Restr, betastr]) + '.dat'
            try:
                with open(path + 'results/' + fname, 'rb') as fin:
                    data = pickle.load(fin)
            except:
                warnings.warn('careful, file {s} not found!'.format(s=fname))
            x_opt.append(data['x_opt'])
            f_opt.append(data['f_opt'])
            err_u.append(data['err_u'])
            err_dp.append(data['err_dP'])
        Xo.append(x_opt)
        Fo.append(f_opt)
        Eu.append(err_u)
        Edp.append(err_dp)
        ls1 = col.next()+mark.next()+linst.next()
        ms = marksize.next()
        # ls2 = ls1 + '-'
        ax1.plot(Res, err_u, ls1, lw=2, ms=ms)
        ax2.plot(Res, err_dp, ls1, lw=2, ms=ms)
        # ax3.plot(Res, f_opt, ls1, lw=2, ms=ms)
        # ax4.plot(Res, x_opt, ls1, lw=2, ms=ms)
        ax1.set_xlim([min(Res)-200, max(Res)+200])
        ax1.set_xlabel('Reynolds number')
        ax2.set_xlim([min(Res)-200, max(Res)+200])
        ax2.set_xlabel('Reynolds number')
        # ax3.set_xlim([min(Res)-200, max(Res)+200])
        # ax4.set_xlim([min(Res)-200, max(Res)+200])

        plt.show()

    pass


def plot_f_over_x(prefs, Res, betas, methods, Re_scale=1.):
    ''' Plot F over x for each pref_method_Re_beta case specified. '''
    if type(prefs) is not list:
        prefs = [prefs]
    if type(methods) is not list:
        methods = [methods]
    if type(Res) not in (list, np.array):
        Res = [Res]
    if type(betas) not in (list, np.array):
        betas = [betas]

    ifig = fignum()
    fig1 = plt.figure(ifig)
    ax1 = fig1.add_subplot(111)
    ax1.set_title('Function over x')

    col = itertools.cycle(['k', 'c', 'b', 'r', 'm', 'g'])
    mark = itertools.cycle(['o', 'v', 's', '^', 'd', 'x'])
    for (pref, meth, beta, Re) in itertools.product(prefs, methods, betas,
                                                    Res):
        betastr = 'b' + str(int(beta))
        Restr = 'Re' + str(int(Re))
        fname = '_'.join([pref, meth, Restr, betastr]) + '.dat'
        try:
            with open(path + 'results/' + fname, 'rb') as fin:
                data = pickle.load(fin)
        except:
            warnings.warn('careful, file {s} not found!'.format(s=fname))

        x = data['x']
        f = data['f']

        isort = np.argsort(x)
        xsort = x[isort]
        fsort = f[isort]

        ls1 = col.next()+mark.next()+'-'
        ax1.plot(xsort, fsort, ls1, lw=2, ms=10)

    pass


def errors(est, STEint=True):
    ''' Calculate velocity and pressure drop errors.

    Args:
        est     instance of NSEstimator class, after est.estimate()

    Return:
        err_u       error in velocity
        err_dP      error in pressure drop
    '''
    err_u = norm(est.u_meas_opt.vector() -
                 est.uref_meas.vector())/norm(est.uref_meas.vector())
    # u1, p1 = est.pb_est.w.split(deepcopy=True)
    err_dP = abs(est.pressure_drop(est.p_opt) /
                 est.pressure_drop(est.pref) - 1.)

    print('rel error U:  {0}'.format(err_u))
    print('rel error dP: {0}'.format(err_dP))

    err_dP_stei = None
    if STEint:
        err_dP_stei = abs(est.STEint() / est.pressure_drop(est.pref) - 1.)
        print('rel error STEint: {0}'.format(err_dP_stei))
    return err_u, err_dP, err_dP_stei


def setup1_STEint(prefix):
    Dh = 2*1.0
    Re_in = np.array([500, 1000, 1500, 2500, 3500])  # 3500
    mu = 0.035
    U_inflow = Re_in/Dh*mu
    beta = 200
    meth = 'STEint'

    err_dP_list = []

    for U in U_inflow:
        # update settings
        est = NSEstimator(path + 'input/' + prefix + '_ref.yaml',
                          path + 'input/' + prefix + '_est.yaml')
        est.pb_ref.options['boundary_conditions'][0]['value']['U'] = U
        est.pb_est.options['boundary_conditions'][0]['value']['U'] = U
        est.pb_est.options['estimation']['method'] = meth
        est.pb_est.options['nitsche']['beta1'] = beta

        dP = est.STEint_standalone()
        err_dP = abs(dP / est.pressure_drop(est.pref) - 1.)
        err_dP_list.append(err_dP)

        print('dP_stei:\t{0}'.format(dP))
        print('rel err:\t{0}'.format(err_dP))

        # check if convergence possible from zero init
        saved_data = {
            'err_dP_stei': err_dP,
        }

        Re_loc = int(round(2.*U/0.035))
        out = ('_'.join([prefix, meth, 'Re'+str(Re_loc), 'b'+str(beta)]) +
               '.dat')
        with open(path+'results/'+out, 'wb') as fout:
            pickle.dump(saved_data, fout, protocol=pickle.HIGHEST_PROTOCOL)

    return Re_in, err_dP_list


def setup1(prefix, methods='BFGS', Re_in=None, beta_list=None,
           noslip=False, beta=False, outfile_pref=''):
    ''' compute setup 1 (see
    experiments/parameter_estimation_sensitivity/plan.md)
    The prefix has to match with the input/*_est,ref.yaml files.

    Args:
        prefix (str)                        file name prefix
    Optional Args:
        methods (str, list of str)          optimization method(s)
        Re_in (float, list, ndarray)        Reynolds numbers
        beta_list (float, list, ndarray)    Reynolds numbers
        noslip (bool)                       bool switch: noslip optimization?
        beta (bool)                         bool switch: beta optimization?
        outfile_pref (str)                  prefix for output files

    Return:
        everything..

    '''

    if type(methods) is not list:
        methods = [methods]
    # if prefix is None:
    #     this should never happen!
    #     random = ''.join([random.choice(string.ascii_letters +
    #                                     string.digits) for n in xrange(4)])
    #     prefix = 's1_' + random

    Dh = 2*1.0
    if Re_in is None:
        Re_in = np.array([500, 1000, 1500, 2500, 3500])  # 3500
    elif type(Re_in) in (int, float):
        Re_in = np.array([Re_in])

    if beta_list is None:
        beta_list = np.arange(200, 1000 + 1, 2000)
    elif type(beta_list) in (int, float):
        beta_list = np.array([beta_list])

    Re_in = np.array(Re_in)
    beta_list = np.array(beta_list)

    mu = 0.035
    U_inflow = Re_in/Dh*mu

    if noslip or beta:
        beta_list = [200]

    if outfile_pref:
        outfile_pref += '_'

    #  x0 = np.array(x0)    # s1_BFGS data set was computed with x0=-5.
    results = []
    f_opt = []
    x_opt = []
    BOlist = []
    F = []
    X = []
    err_u_list = []
    err_dP_list = []
    for (meth, beta, U) in itertools.product(methods, beta_list, U_inflow):
        print(meth, beta, U)
        # update settings
        est = NSEstimator(path + 'input/' + prefix + '_ref.yaml',
                          path + 'input/' + prefix + '_est.yaml')
        est.pb_ref.options['boundary_conditions'][0]['value']['U'] = U
        est.pb_est.options['boundary_conditions'][0]['value']['U'] = U
        est.pb_est.options['estimation']['method'] = meth
        est.pb_est.options['nitsche']['beta1'] = beta
        est.options = copy.deepcopy(est.pb_est.options)

        est.estimate()

        results.append(est.result)
        f_opt.append(est.f_opt)
        x_opt.append(est.x_opt)
        BOlist.append(est.BO)
        F.append(est.fval)
        X.append(est.x)

        # check if convergence possible from zero init
        est.solve_opt(est.x_opt, init=True)

        err_u, err_dP, err_dP_stei = errors(est)
        err_u_list.append(err_u)
        err_dP_list.append(err_dP)

        saved_data = {
            'x': est.x,
            'f': est.fval,
            'f_opt': est.f_opt,
            'x_opt': est.x_opt,
            'err_u': err_u,
            'err_dP': err_dP,
            'err_dP_stei': err_dP_stei,
            'U': U,
            'beta': beta,
            'method': meth
        }

        Re_loc = int(round(2.*U/0.035))
        out = ('_'.join([prefix, meth, 'Re'+str(Re_loc), 'b'+str(beta)]) +
               '.dat')
        with open(path+'results/'+outfile_pref+out, 'wb') as fout:
            pickle.dump(saved_data, fout, protocol=pickle.HIGHEST_PROTOCOL)

    outfile = '_'.join([prefix] + methods) + '.dat'
    alldata = {
        'results': results,
        'x': X,
        'f': F,
        'f_opt': f_opt,
        'x_opt': x_opt,
        'err_u': err_u_list,
        'err_dP': err_dP_list,
        'Re': Re_in,
        'beta': beta_list,
        'method': methods
    }
    with open(path+'results/'+outfile_pref+outfile, 'wb') as fout:
        pickle.dump(alldata, fout, protocol=pickle.HIGHEST_PROTOCOL)

    return (results, f_opt, x_opt, BOlist, F, X, err_u_list, err_dP_list,
            Re_in, beta_list, methods)


def compute_velocity_measurement(prefixes, Re_in=None):
    ''' Compute U_meas for prefix and export to VTK.

    Args:
        prefixes            str or list of prefixes
        Re_in (optional)    single or various Reynolds numbers
    '''
    Dh = 2*1.0
    mu = 0.035
    if not Re_in:
        Re_in = np.array([500, 1000, 1500, 2500, 3500])  # 3500

    if type(Re_in) not in (list, tuple):
        Re_in = [Re_in]

    if type(prefixes) not in (list, tuple):
        prefixes = [prefixes]

    for prefix, Re in itertools.product(prefixes, Re_in):
        U = Re/Dh*mu
        est = NSEstimator(path + 'input/' + prefix + '_ref.yaml',
                          path + 'input/' + prefix + '_est.yaml')
        est.pb_ref.options['boundary_conditions'][0]['value']['U'] = U

        print('computing U = {0}'.format(U))
        est.measurement()

        fname = prefix + '_Re' + str(Re) + '_u_meas.pvd'
        print('writing file %path/results/' + fname)
        f1 = File(path + 'results/' + fname)
        est.uref_meas.rename('u0', 'u_measurement')
        f1 << est.uref_meas

    pass


def fignum():
    ''' return max fig number +1 '''
    return max(plt.get_fignums()) + 1 if plt.get_fignums() else 1


def scatter_param(E, subplot=False):
    ''' make scatter plot for 4 parameters of one or several estimations.'''
    plt.ion()
    if type(E) not in (list, tuple):
        E = [E]
    fig_num = max(plt.get_fignums()) + 1 if plt.get_fignums() else 1
    for est in E:
        if est.options['estimation']['method'] == 'gpyopt':
            X = est.BO.X
            F = est.BO.Y
        else:
            X = est.x
            F = est.fval
        Ns = X.shape[1]

        if subplot and Ns == 4:
            fig, ax = plt.subplots(2, 2, sharey='row')
            print(ax)
            ax[0, 0].scatter(X[:, 0], F)
            ax[0, 0].set_xlabel(r'$x_{i}$'.format(i=0))
            ax[0, 0].set_ylabel(r'$F(x_i)$')
            ax[0, 1].scatter(X[:, 1], F)
            ax[0, 1].set_xlabel(r'$x_{i}$'.format(i=1))
            ax[1, 0].scatter(X[:, 2], F)
            ax[1, 0].set_xlabel(r'$x_{i}$'.format(i=2))
            ax[1, 0].set_ylabel(r'$F(x_i)$')
            ax[1, 1].scatter(X[:, 3], F)
            ax[1, 1].set_xlabel(r'$x_{i}$'.format(i=3))

        else:
            for i in range(Ns):
                plt.figure(fig_num + i)
                plt.scatter(X[:, i], F)
                plt.xlabel(r'$x_{i}$'.format(i=i))
                plt.ylabel(r'$F(x_i)$')

    pass


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
