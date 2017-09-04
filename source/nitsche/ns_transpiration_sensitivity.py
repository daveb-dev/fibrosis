from dolfin import *
#  import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import os
#  from mpl_toolkits.mplot3d import Axes3D
#  from matplotlib2tikz import save as tikz_save
#  from solvers.ns_steady.nssolver import NSSolver
#  from solvers.ns_steady.nsproblem import NSProblem
from solvers.ns_steady.nsestimator import NSEstimator

#  ''' optimization flags
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
#  parameters["form_compiler"]["representation"] = "quadrature"
parameters["form_compiler"]["cpp_optimize_flags"] = \
    "-O3 -ffast-math -march=native"

#  '''
path = './experiments/transpiration_sensitivity/'
plt.ion()


def compute(prefix, out_pref='', Re=None, seed=None):
    estfile = path + 'input/' + prefix + '_est.yaml'
    reffile = path + 'input/' + prefix + '_ref.yaml'
    est = NSEstimator(reffile, estfile)

    if seed is not None:
        est.pb_est.options['estimation']['random_seed'] = seed
        out_pref += ('_' if out_pref else '') + 'seed{}'.format(seed)

    est.estimate()
    est.solve_opt(est.x_opt, init=True)

    err_u, err_dP, err_dP_stei, err_dP_ste, err_dP_ppe, du_field = \
        errors(est, STEint=True, du_meas=True)

    err_u, err_dP, dP, dP_ref, du_field, dP_direct, err_dP_direct = \
        pressure_drop(est, STEint=True, du_meas=True)

    dP_stei, dP_ste, dP_ppe = dP_direct
    err_dP_stei, err_dP_ste, err_dP_ppe = err_dP_direct

    # write results
    if not Re:
        U = est.pb_ref.options['boundary_conditions'][0]['value']['U']
        R = est.pb_ref.options['boundary_conditions'][0]['value']['R']
        mu = est.pb_ref.options['mu']
        Re = int(round(2.*R*U/mu))
        print('Reynolds number:  {0}'.format(Re))

    # python pickle data
    save_pickle = {
        'x': est.x,
        'f': est.fval,
        'f_opt': est.f_opt,
        'x_opt': est.x_opt,
        'x_leg': est.xlegend,
        'err_u': err_u,
        'dP_ref': dP_ref,
        'err_dP': err_dP,
        'dP': dP,
        'err_dP_stei': err_dP_stei,
        'err_dP_ste': err_dP_ste,
        'err_dP_ppe': err_dP_ppe,
        'dP_stei': dP_stei,
        'dP_ste': dP_ste,
        'dP_ppe': dP_ppe
    }

    # make results dir for current case
    resdir = path + 'results/' + prefix + ('' if not out_pref else '_' +
                                           out_pref) + '/'
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    if 'save_data' in est.pb_ref.options and est.pb_ref.options['save_data']:
        est.pb_ref.save_HDF5()

    XDMFFile(resdir + '/u_opt_{Re}.xdmf'.format(Re=Re)).write(est.u_opt)
    XDMFFile(resdir + '/p_opt_{Re}.xdmf'.format(Re=Re)).write(est.p_opt)
    XDMFFile(resdir + '/u_ref_{Re}.xdmf'.format(Re=Re)).write(est.uref)
    XDMFFile(resdir + '/p_ref_{Re}.xdmf'.format(Re=Re)).write(est.pref)
    XDMFFile(resdir + '/u_ref_meas_{Re}.xdmf'.format(Re=Re)).write(
        est.uref_meas)
    XDMFFile(resdir + '/u_diff_{Re}.xdmf'.format(Re=Re)).write(du_field)
    # write u_meas ?

    with open(resdir + '/estim_results.dat', 'wb') as fout:
        pickle.dump(save_pickle, fout, protocol=pickle.HIGHEST_PROTOCOL)

    return save_pickle


def pressure_drop(est, STEint=True, du_meas=True):
    ''' Calculate velocity and pressure drop errors.

    # REPLACES errors()

    Args:
        est     instance of NSEstimator class, after est.estimate()

    Return:
        err_u       error in velocity
        err_dP      error in pressure drop
    '''
    dP = est.pressure_drop(est.p_opt)
    dP_ref = est.pressure_drop(est.pref)

    err_u = norm(est.u_meas_opt.vector() -
                 est.uref_meas.vector())/norm(est.uref_meas.vector())
    # u1, p1 = est.pb_est.w.split(deepcopy=True)
    err_dP = abs(dP/dP_ref - 1.)

    print('rel error U:  {0}'.format(err_u))
    print('rel error dP: {0}'.format(err_dP))

    dP_direct = [0]*3
    err_dP_direct = [0]*3
    if STEint:
        dP_stei = est.STEint()[0]
        dP_ste = est.STE()[0]
        dP_ppe = est.PPE()[0]
        err_dP_stei = abs(dP_stei/dP_ref - 1.)
        err_dP_ste = abs(dP_ste/dP_ref - 1.)
        err_dP_ppe = abs(dP_ppe/dP_ref - 1.)
        print('rel error STEint: {0}'.format(err_dP_stei))
        print('rel error STE:    {0}'.format(err_dP_ste))
        print('rel error PPE:    {0}'.format(err_dP_ppe))
        dP_direct = (dP_stei, dP_ste, dP_ppe)
        err_dP_direct = (err_dP_stei, err_dP_ste, err_dP_ppe)

    du_field = None
    if du_meas:
        u = est.u_meas_opt
        uref = est.uref_meas
        du_field = Function(u.function_space())
        du_field.assign(u-uref)

    return err_u, err_dP, dP, dP_ref, du_field, dP_direct, err_dP_direct


def errors(est, STEint=True, du_meas=True):
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
        dP_ref = est.pressure_drop(est.pref)
        err_dP_stei = abs(est.STEint()[0]/dP_ref - 1.)
        err_dP_ste = abs(est.STE()[0]/dP_ref - 1.)
        err_dP_ppe = abs(est.PPE()[0]/dP_ref - 1.)
        print('rel error STEint: {0}'.format(err_dP_stei))
        print('rel error STE:    {0}'.format(err_dP_ste))
        print('rel error PPE:    {0}'.format(err_dP_ppe))

    du_field = None
    if du_meas:
        u = est.u_meas_opt
        uref = est.uref_meas
        du_field = Function(u.function_space())
        du_field.assign(u-uref)

    return err_u, err_dP, err_dP_stei, err_dP_ste, err_dP_ppe, du_field


def prepend_(s):
    ''' prepend underscore to string if not none. '''
    return '_' + s if s else ''


def simulate_paramset(params, out_pref='', seed=None):
    ''' simulate all combination of parameters. '''
    results = []
    f0lst, gmlst, noiselst = params
    for (f0, gm, noise) in itertools.product(f0lst, gmlst, noiselst):
        pref = 's1{f0}{gm}{noise}'.format(f0=prepend_(f0), gm=prepend_(gm),
                                          noise=prepend_(noise))
        print(pref)
        results.append(compute(pref, out_pref=out_pref, seed=seed))

    return results


if __name__ == '__main__':

    def simulate_beta_grid():
        f0 = [0.0, 0.2, 0.4, 0.6]
        beta = [0, 500, 1000, 1500, 2000]
        results = []
        for f, b in itertools.product(f0, beta):
            pref = 's2/f{f}/s2_f{f}_beta{b}'.format(f=f, b=b)
            print(pref)
            results.append(compute(pref))
        return results

    def simulate_gamma_grid():
        f0 = [0.0, 0.2, 0.4, 0.6]
        gamma = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = []
        for f, b in itertools.product(f0, gamma):
            pref = 's2/f{f}/s2_f{f}_gamma{b}'.format(f=f, b=b)
            print(pref)
            results.append(compute(pref))
        return results

    def bruteforce():
        f0lst = [0.0, 0.2, 0.4, 0.6]

        results = []
        for f0 in f0lst:
            pref = 's2/f{f}/s2_f{f}_bruteforce'.format(f=f0)
            print(pref)
            results.append(compute(pref))
        return results

    def main1():
        results = []

        # set 1
        f0lst = ['f0.6', 'f0.4', 'f0.2']
        #  gmlst = ['', 'gm-const', 'gm0']
        gmlst = ['']
        noiselst = ['', 'noise']
        #  results += simulate_paramset((f0lst, gmlst, noiselst))

        # set 2
        f0lst = ['f0.6']
        noiselst = ['', 'noise']
        gmlst = ['gm-const', 'gm0']
        #  simulate_paramset((f0lst, gmlst, noiselst))

        # set 2: noslip
        #  f0lst = ['f0.2', 'f0.4', 'f0.6']
        f0lst = ['f0.2', 'f0.4']
        noiselst = ['', 'noise']
        gmlst = ['noslip']
        results += simulate_paramset((f0lst, gmlst, noiselst), seed=10)

        # additionally
        extra = [{'pref': 's1_bl_f0.6'}, {'pref': 's1_bl_f0.6_noise'}]
        for ex in extra:
            print(ex['pref'])
            #  results.append(compute(ex['pref']))
        return results

    def main2():
        results = []
        preflst = ['s1_f0.6', 's1_f0.6_noslip']
        for pref in preflst:
            results.append(compute(pref))
        return results

    def main3():
        results = []
        preflst = ['s1_bl_f0.6_L6', 's1_bl_f0.6_L6_noise']
        # 's1_f0.6_gm_direct']
        for pref in preflst:
            results.append(compute(pref))
        return results

    def main_gmnoise():
        results = []
        f0lst = ['f0.6']
        noiselst = ['noise']
        gmlst = ['', 'gm-const', 'gm0']
        for i in range(3, 11):
            results += simulate_paramset((f0lst, gmlst, noiselst),
                                         out_pref=str(i))
        return results

    def main_f06noise_noslip_SEED(seeds=range(100)):
        results = []
        f0lst = ['f0.6']
        noiselst = ['noise']
        gmlst = ['noslip']
        for i in seeds:
            results += simulate_paramset((f0lst, gmlst, noiselst), seed=i)
        return results

    def main_f06noise_slip_SEED(seeds=range(100)):
        results = []
        f0lst = ['f0.6']
        noiselst = ['noise']
        gmlst = ['']
        for i in seeds:
            results += simulate_paramset((f0lst, gmlst, noiselst), seed=i)
        return results

    def main_gmnoise_SEED():
        results = []
        f0lst = ['f0.6']
        noiselst = ['noise']
        gmlst = ['gm-const', 'gm0']
        seeds = [10, 11]
        for seed in seeds:
            results += simulate_paramset((f0lst, gmlst, noiselst),
                                         seed=seed)

        return results

    ###############

    #  results = main_gmnoise_SEED()
    #  main2()
    #  res1 = simulate_gamma_grid()
    #  res2 = simulate_beta_grid()
    #  res1 = bruteforce()
    results = main_f06noise_noslip_SEED(seeds=range(75, 100))
    results = main_f06noise_slip_SEED(seeds=range(75, 100))
    pass
