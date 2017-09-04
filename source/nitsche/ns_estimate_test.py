''' Testing script/playground for NSEstimator '''
from dolfin import *
from solvers.ns_steady.nsestimator import NSEstimator

#  ''' optimization flags
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
#  parameters["form_compiler"]["representation"] = "quadrature"
parameters["form_compiler"]["cpp_optimize_flags"] = \
    "-O3 -ffast-math -march=native"


path = './experiments/transpiration_sensitivity/input/'
#  path = './input/'

opt_file = path + 's1_f0.6_noise_est.yaml'
ref_file = path + 's1_f0.6_noise_ref.yaml'
est = NSEstimator(ref_file, opt_file)
est.pb_est.options['estimation']['random_seed'] = 10

est.estimate()
est.solve_opt(est.x_opt, init=True)


err_u = norm(est.u_meas_opt.vector() -
             est.uref_meas.vector())/norm(est.uref_meas.vector())
# u1, p1 = est.pb_est.w.split(deepcopy=True)
err_dP = abs(est.pressure_drop(est.p_opt) /
             est.pressure_drop(est.pref) - 1.)
err_dP_stei = abs(est.STEint() / est.pressure_drop(est.pref) - 1.)

print('rel error U:  {0}'.format(err_u))
print('rel error dP: {0}'.format(err_dP))
print('rel error STEint: {0}'.format(err_dP_stei))


# extra: compute U error field
u = est.u_meas_opt
uref = est.uref_meas
du_field = Function(u.function_space())
du_field.assign(u-uref)
du_field.rename('u_diff', 'u_diff')
