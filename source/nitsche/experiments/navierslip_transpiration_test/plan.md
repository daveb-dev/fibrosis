# Navier-slip and Transpiration BC

- test the new transpiration boundary condition
- compare Nitsche vs. Transpiration
- analyze resistance coefficient (beta)

## Comparison Nitsche vs Transpiration

### Scenarios:
- `s1c`: $\Delta R = 0.1$, $Re = 3500$ ($U = 61.25$),
    meshes: `coarc2d_f0.6_dX_ns1_hY.h5`, $X\in[0,0.1]$, $Y\in[0.05,0.025]$, meas: $Y=0.1$

- `s2c`: $\Delta R = 0.1$, $Re = 3500$ ($U = 61.25$)
    mesh: `coarc2d_sc_f0.6_d0.1_ns1_h0.05.h5`
     ref: `coarc2d_bl_sc_f0.6_d0_ns1_h0.025.h5` (boundary layer refinement)
    meas: `coarc2d_sc_f0.6_d0.1_ns1_h0.1.h5`

- `_noise`: $+ 10$% Gaussian noise, analyze average of 10 (?) realizations

### Parameters:
- __Nitsche__:  (3 parameters, $\beta=200$ fixed)   `_no-pen`
    +  optimize `inflow`  ($U$ (lin), $\Delta R$ (exp))
    a) optimize `navierslip`  ($\gamma=const$ via $\Delta R$ (exp))
    b) optimize `navierslip`  ($\gamma=\gamma(x)$ via $\Delta R$ (exp))

- __Transpiration__:  (4 parameters)            `_transp`
    +  optimize `inflow`  ($U$ (lin), $\Delta R$ (exp))     [essential]
    a) optimize `navierslip`  ($\gamma=const$ via $\Delta R$ (exp))
    b) optimize `navierslip`  ($\gamma=\gamma(x)$ via $\Delta R$ (exp))
    c) take $\Delta R \to \infty$ or $\gamma = 0$
    +  optimize `transpiration` ($\beta$ (exp))

### Visualization:
- Bar plot errors ($U$, $\delta P$)
    for [np, transp, avg(np+noise), avg(transp+noise)] @ $Re = 3500$

