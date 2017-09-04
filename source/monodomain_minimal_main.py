'''   Solve Monodomain Equations with the Minimal ventricular model (Bueno-Ovorio) for reaction term, i.e:
	
	dot u - div ( D grad u ) = f(phi,r,w,s)   
	dot r = f1(phi, r)
	dot w = f2(phi, w)
	dot s = f3(phi)
	
	+ No-flux conditios over all boundaries
	
	Remember to use instant-clean to clean up python cache
	
	Parallel: mpirun -np 8 python monodomain_minimal_main.py	
'''

if __name__ == "__main__":
    from functions import inout as io
    from functions import utils
    from solvers.monodomain_minimal import compute
    import time

    # Load simulation parameters from yaml file
    parameter_file = './input/prms_monodomain_minimal.yaml'
    prms = io.prms_load(parameter_file)

    # Make results directory if applicable
    utils.trymkdir(prms['io']['results'])

    # Print parameters on 1st proc only
    io.prms_print(prms)
    
    compute(prms)


