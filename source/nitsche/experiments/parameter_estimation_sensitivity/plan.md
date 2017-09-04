# Experimentation plan
using FEniCS 1.6.0 (Hashstack) and `ns_steady origin/nitsche_estim_newton`
as of commit @abbc39b 
`ns_sensitivity.py` (see `./src_backup`, @blob)
input files: `coarc_(ref/estim)_exp_sens.yaml`

Setup 1): 1 parameter
a) analyze estimation process
    - optimize $f(\gamma)$ for several $Re$ and $\beta$s
    - optimize $f(\gamma, \beta)$ for several $Re$   (3D scatter?)
    - check $\beta(Re)$ limits, choose ideal $\beta$ $\to$ can $\beta$ be
      chosen constant for all simulations or strong dependency on $Re$?
    - plot $f_{opt}$ over $Re$ for several $\beta$s
    - plot $f_{opt}(\gamma_{opt}, \beta_i)$ over $\beta_i$ for 3 $Re$
    - sensitivity of $f$: plot $f$ over $\gamma_i$ for several $Re$, with $\beta$ fixed
    - compare $f_{opt}$ obtained with BFGS/Nelder-Mead (L-BFGS?) with GPyOpt
    - DO THIS FOR $dR = 0.05$ and $dR = 0.1$ !!
    - DO THIS FOR NO-SLIP
    - ADD NOISE
b) results
    - plot $\mathrm{err}(u), \mathrm{err}(\delta p)$ over $Re$ (for $\beta$s?)
    - compare to no-slip, STEint(?)

Datasets: `./results/s1_*.dat` --> $dR=0.05$
:         `./results/s1b_*.dat` --> $dR=0.1$
:         `./results/s1_noslip_*.dat` --> $dR=0.05$, no-slip
:         `./results/s1b_noslip_*.dat` --> $dR=0.1$, no-slip
:         `./results/s1*_noise_*.dat` --> NOISE
:         `./results/s1c_*.dat` --> $H_{meas} = dR = 0.1$
:         `./results/s1c_noise_*.dat` --> NOISE and $H_{meas} = dR = 0.1$
NOTE: The BFGS `s1` calculations were initialized with $x0 = -5.$,
      GPyOpt with $x0 = [[-6.2], [-5.]]$ (otherwise divergence)


Setup 2): 2 parameters
a) analyze estimation process
    - optimize with 2 parameters, several $\beta$s and $Re$s
    - plot $f_{opt}$ over $Re$ for several $\beta$s
    - sensitivity: 
        - plot $f$ over $(\gamma_1, \gamma_2)$ (3D scatter)
        - make individual scatter plots $f$ over $\gamma_i$
        - OAT (one at at time): fix one $\gamma$ (on a good value) and vary the
          other (include $\beta$ here?)
    - compare $f_{opt}(\gamma)$ with $f_{opt}(\gamma_1, \gamma_2)$
b) results:
    - if new insights are gained?


Setup 3): 4 parameters
a) analyze estimation process
    - optimize $f$ over all $\gamma$s for different $(\beta, Re)$
    - compare $f_{opt}$ obtained with ${1; 2; 4}$ parameters
    - sensitivity:
        - $\forall i$, change one parameter at a time, keep others fixed
        - scatter plots $f(x_i), i=1,...,N$ over $x_j$
    - compare optimization methods
b) results:
    - depending on a)

    

        
    
