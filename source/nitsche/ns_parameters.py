from dolfin import *
# from solvers.ns_steady.nssolver import NSSolver
# from solvers.ns_steady.nsproblem import NSProblem
from solvers.ns_steady.paramstudy import ParameterStudy
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

# inputfile = 'input/cavity.yaml'
# inputfile = 'input/bfs.yaml'
inputfile = 'input/coarc.yaml'
# inputfile = 'input/pipe.yaml'

param_modlist = [
    {'nonlinear': {'method': 'picard', 'use_aitken': 0}},
    {'nonlinear': {'method': 'picard', 'use_aitken': 1}},
    # {'nonlinear': {'method': 'snes'}},
    {'nonlinear': {'method': 'newton', 'use_aitken': 1}},
    # {'nonlinear': {'method': 'qnewton'}}
]
legend = ('Picard', 'Picard + Aitken', 'Newton')

ps = ParameterStudy(inputfile, param_modlist)
ps.solve()

# for dimensionless case
Re = int(round(1./ps.options['mu']))
print('Re = {0}'.format(Re))

plt.ion()
plt.figure()
for resid in ps.residuals:
    plt.semilogy(resid)
plt.legend(legend)
plt.grid()
plt.xlabel('iterations')
plt.ylabel(r'$||R_k||_{L_2}$')
tikz_save(
    'results/ns_bench/coarc_resid_Re{0}.tikz'.format(Re),
    figureheight='\\figureheight',
    figurewidth='\\figurewidth'
)

plt.figure()
for energy in ps.energy:
    plt.plot(energy, '-o')
plt.grid()
plt.xlabel('iterations')
plt.ylabel(r'$\mathcal{E}_k$')
plt.legend(legend, loc=4)
tikz_save(
    'results/ns_bench/coarc_energy_Re{0}.tikz'.format(Re),
    figureheight='\\figureheight',
    figurewidth='\\figurewidth'
)
