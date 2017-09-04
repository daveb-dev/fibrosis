''' Testing script/playground for NSEstimator '''
from dolfin import *
from solvers.ns_steady.nsestimator import NSEstimator
import numpy as np

#  ''' optimization flags
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
#  parameters["form_compiler"]["representation"] = "quadrature"
parameters["form_compiler"]["cpp_optimize_flags"] = \
    "-O3 -ffast-math -march=native"

set_log_level(30)
set_log_active(False)

path = './experiments/transpiration_sensitivity/input/'
#  path = './input/'

noise = True
methods = ['PPE', 'STE', 'STEint']

if not noise:
    #  suf = 'd0_H0.05_'
    #  suf = 'd0_H0.1_'
    #  suf = ''
    #  suf = 'd0.1_H0.05_'
    suflst = ['d0_H0.05_', 'd0_H0.1_', 'd0.1_H0.05_', '']
    err_dP = []
    dP = []
    for suf in suflst:
        opt_file = path + 's1_f0.6_{}est.yaml'.format(suf)
        ref_file = path + 's1_f0.6_ref.yaml'
        est = NSEstimator(ref_file, opt_file)

        est.measurement()
        dP_ref = est.pressure_drop(est.pref)

        dP_loc = []
        err_dP_loc = []
        for meth in methods:
            dP_, p_est = est.direct_estimator(meth, return_pressure=True)
            dP_loc.append(dP_)
            err_dP_loc.append(abs(dP_/dP_ref - 1))
            XDMFFile(meth + '_' + suf + 'p.xdmf').write(p_est)

        dP.append(dP_loc)
        err_dP.append(err_dP_loc)

        print('\nPressure drop REF:\t{}'.format(dP_ref))
        for meth, dP_, err in zip(methods, dP_loc, err_dP_loc):
            print('Pressure drop {0}:\t{1:.2f}  (error:  {2:.3f})'.format(
                meth, dP_, err))

if noise:
    #  suf = 'd0_H0.05_'
    #  suf = 'd0_H0.1_'
    #  suf = 'd0.1_H0.05_'
    suflst = ['d0.1_H0.05_', 'd0_H0.1_', 'd0_H0.05_', '']
    for suf in suflst:
        opt_file = path + 's1_f0.6_{}noise_est.yaml'.format(suf)
        ref_file = path + 's1_f0.6_noise_ref.yaml'
        est = NSEstimator(ref_file, opt_file)

        dP = []
        err_dP = []

        seeds = range(200)
        for s in seeds:
            est.pb_est.options['estimation']['random_seed'] = s

            est.measurement()
            dP_ref = est.pressure_drop(est.pref)

            for meth in methods:
                dP.append(est.direct_estimator(meth))
                err_dP.append(abs(dP[-1]/dP_ref - 1))
                print('Pressure drop {0}:\t{1:.2f}  (error:  {2:.3f})'.format(
                    meth, dP[-1], err_dP[-1]))

        res = np.array(err_dP).reshape((len(seeds), len(methods)))
        dP = np.array(dP).reshape((len(seeds), len(methods)))

        print('\nPressure drop REF:\t{}'.format(dP_ref))
        for i, meth in enumerate(methods):
            print('{}:'.format(meth))
            print('   dP mean:     {0}'.format(dP[:, i].mean()))
            print('   error mean:  {0}'.format(res[:, i].mean()))
            print('   error std:   {0}'.format(res[:, i].std()))

            np.savez('errors_{0}_{1}noise_200.npz'.format(meth, suf),
                     dP=dP[:, i], err=res[:, i], mean=res[:, i].mean(),
                     std=res[:, i].std(), dP_ref=dP_ref)
