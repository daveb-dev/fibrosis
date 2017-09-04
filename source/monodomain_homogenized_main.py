if __name__ == "__main__":

    from dolfin import *
    from functions import inout as io
    from functions import utils
    from solvers.monodomain_homogenized import compute

    parameter_file = './input/prms_monodomain_homogenized.yaml'

    prms = io.prms_load(parameter_file)

    # Make results directory if applicable
    utils.trymkdir(prms['io']['results'])

    # Print parameters on 1st proc only
    io.prms_print(prms)

    import time

    t0 = time.clock()
    compute(prms)
    tt = time.clock() - t0
    print "\n\n --- End of Simulation (total time elapsed: %g [seg]) ---" % tt
