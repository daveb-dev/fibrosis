# Navier-slip and Transpiration BC

- test the new transpiration boundary condition
- compare Nitsche vs. Transpiration
- analyze resistance coefficient (beta)

## Transpiration: severeness of coarctation

### Scenarios:
- `s1_f0.6`: $\Delta R = 0.1$, $Re = 3500$ ($U = 61.25$),
    meshes: `coarc2d_Lc2_L5_f0.6_dX_ns1_hY.h5`, $X\in[0,0.1]$, $Y\in[0.05,0.025]$, meas: $Y=0.1$
- `s1_bl_f0.6`: $\Delta R = 0.1$, $Re = 3500$ ($U = 61.25$),
    meshes: `coarc2d_bl_Lc2_L5_f0.6_dX_ns1_hY.h5`, $X\in[0,0.1]$, $Y\in[0.05,0.025]$, meas: $Y=0.1$
    - __with boundary layer__

- `s1_f0.4`, `s1_f0.2` similarly

<!-- - `s2_f0.6`: $\Delta R = 0.1$, $Re = 3500$ ($U = 61.25$) -->
<!--     mesh: `coarc2d_Lc1_L5_f0.6_d0.1_ns1_h0.05.h5` -->
<!--      ref: `coarc2d_bl_Lc1_L5_f0.6_d0_ns1_h0.025.h5` (boundary layer refinement) -->
<!--     meas: `coarc2d_Lc1_L5_f0.6_d0.1_ns1_h0.1.h5` -->
<!--     - **coarctation shorter & steeper** -->

- `_noise`: $+ 10$% Gaussian noise, analyze average of 100 (?) realizations

NEW 07/10/2016:
- `s2`: `input/s2/f0.x` 
    - compute $\beta \times \gamma$ grid
    - inlet fixed $U=60$, $\delta R=0.1$
    - $\beta \in (0, 500, 1000, 1500, 2000)$
    - $\gamma \in (0, 0.25, 0.5, 0.75, 1.0)$
    - input files: `./input/s2/f0.x/s2_f0.x_gamma|betaY_est|ref.yaml`

**NOTE**: this uses solver version v0.1 and FEniCS 2016.1
    - $f_0=0.6$, noslip/navierslip and PPE/STE(int) recalculated with v0.1.3


### Parameters:
<!-- - __Nitsche__:  (3 parameters, $\beta=200$ fixed)   `_no-pen` -->
<!--     +  optimize `inflow`  ($U$ (lin), $\Delta R$ (exp)) -->
<!--     a) optimize `navierslip`  ($\gamma=const$ via $\Delta R$ (exp)) -->
<!--     b) optimize `navierslip`  ($\gamma=\gamma(x)$ via $\Delta R$ (exp)) -->

- __Transpiration__:  (4 parameters)            `_transp`
    +  optimize `inflow`  ($U$ (lin), $\Delta R$ (exp))     [essential]
    a) optimize `navierslip`  ($\gamma=\gamma(x)$ via $\Delta R$ (exp))
    b) optimize `navierslip`  ($\gamma=const$ via $\Delta R$ (exp))
        - > `sX_gm-const`
    c) take $\Delta R \to \infty$ or $\gamma = 0$
        - > `sX_gm0`
    +  optimize `transpiration` ($\beta$ (exp))

    --> simulations:
        - $\forall$      $f_0=0.2, 0.4, 0.6$,   ($Lc=1, 2$),    $noise=0, 0.1$
:                $\gamma = 0, const, f(x)$
:                $3 x 2 x 3 (x2) = 18 (36)$    minimization problems
:                time ~4min =>   total ~1h    
            

### Visualization
3D plots:  error vs ($\beta \times \gamma$)   $\forall$ realizations
    - influence of noise 
    - influence of $\gamma$ def (const vs local)
    - compare $\gamma,\beta$ sensitivity wrt to $f_0$ with optimal $\gamma$ choice and with/without noise

