from nsproblem import NSProblem
from nssolver import NSSolver
import collections


class ParameterStudy:
    ''' Perform parameter study for NSSolver and NSProblem, based on the
    parameters specified via the input yaml-file.
    A list of dictionaries is given, containing sets of parameters to be
    modified with respect to the base option set.
    '''
    def __init__(self, inputfile, param_list):
        ''' Initialize with input file.

        Args:
            inputfile   path to input.yaml
            param_list  list of dictionaries with parameter updates
        '''
        self.options = []
        self.inputfile = inputfile
        self.param_list = param_list
        self.residuals = []
        self.energy = []
        self.sol = []
        pass

    def _solve_paramset(self, parameters):
        ''' Setup problem and solver for modified parameters, then solve.

        Args:
            parameters      dict with parameter update
        '''
        problem = NSProblem(self.inputfile)

        if parameters:
            self.update(problem.options, parameters)
            print(parameters)

        self.options = problem.options

        problem.init()
        solver = NSSolver(problem)
        solver.solve()

        self.sol.append(solver.w)
        self.residuals.append(solver.residual)
        self.energy.append(solver.energy)
        pass

    def solve(self):
        ''' Solve for all parameter sets. First solve for the parameters given
        in the input file. Then loop over all given update dicts.
        '''

        # # solve for parameters from input file
        # self._solve_paramset(None)
        # FIXME: don't solve for input file setup !

        for param in self.param_list:
            self._solve_paramset(param)

        pass

    def update(self, d, u):
        ''' Update nested dictionaries of arbitrary depth recursively,
        cf. http://stackoverflow.com/a/3233356/4817051

        Args:
            d   original dict
            u   update
        '''
        for k, v in u.iteritems():
            if isinstance(v, collections.Mapping):
                r = self.update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        return d
