// https://openfoamwiki.net/index.php/2D_Mesh_Tutorial_using_GMSH
/*
  Profile of the axisymmetric stenosis following a cosine
  function dependent on the axial coordinate x [1].
 
   f(x) = R * (1 - f0/2 *(1 + Cos(2.0*Pi * (x-x0)/L) )
   -- x0 maximum stenosis position.
   -- L stenosis length.
   -- R vessel radius
   -- f0 obstruction fraction, ranging 0--1.
   
  References:
 
  [1] Varghese, S. S., Frankel, S. H., Fischer, P. F., 'Direct numerical
simulation of steotic flows. Part 1. Steady flow', Journal of Fluid Mechanics,
vol. 582, pp. 253 - 280.
*/

// coarc 2d mesh with 1-4 boundary segments
// 1: complete top boundary is one physical group
// 2: top boundary split into two physical groups (or split in x0 or
//    straight/curved section ? FIXME)
// 3: in/curve/out or in+out/curve1/curve2
// 4: in/out/curve1/curve2


// CONFIGURATION
Nseg = 1;   // 1, 2, 3, 4
boundary_layer = 0;   // set 0 for measurement mesh!
h = 0.1;
reflevels = 0;
fname = "coarc2d_";

f0 = 0.6; // [0;1]

R_base = 1.0;  // cm
Xi = 1.0;
Xo = 2.;
L = 2.;      // short: 1.0, sc2: 0.5, normal: 2.0


// create geometry and mesh
x0 = Xi + L/2.0;

len = Xi + L + Xo;

i = -1;
dR[i++] = 0;  // reference mesh
// dR[i++] = 0.05;
// dR[i++] = 0.1;
// dR[i++] = 0.15;
// dR[i++] = 0.2;
lastInd = i;

Mesh.CharacteristicLengthFromPoints = 1;

For k In {0:lastInd}
  // Delete physical groups
  Delete Model;

  R = R_base - dR[k];
  /*
    h = 5;
   
    Xi = 100; // um
    Xo = 100; // um
    L = 100.0; // um
   
    R = 50.0;   // um
    f0 = 0.5;   // 0--1
   
    Z = 5;
  */

  p1 = newp; Point(p1) = {0, 0, 0, h};
  // p2 = newp; Point(p2) = {Xi, 0, 0, h};
  p3 = newp; Point(p3) = {Xi, R, 0, h};    // start of cosine
  p4 = newp; Point(p4) = {0, R, 0, h};

  // p5 = newp; Point(p5) = {Xi + L, 0, 0, h};
  p6 = newp; Point(p6) = {Xi + L + Xo, 0, 0, h};
  p7 = newp; Point(p7) = {Xi + L + Xo, R, 0, h};
  p8 = newp; Point(p8) = {Xi + L, R, 0, h};   // end of cosine

  p9 = newp; Point(p9) = {Xi + L/2, R_base*(1.-f0) - dR[k], 0, h};  // middle point

  l0 = newl; Line(l0) = {p1, p6};
  // l1 = newl; Line(l1) = {p1, p2};
  // l2 = newl; Line(l2) = {p2, p3};
  l3 = newl; Line(l3) = {p3, p4};
  l4 = newl; Line(l4) = {p4, p1};

  // l5 = newl; Line(l5) = {p5, p6};
  l6 = newl; Line(l6) = {p6, p7};
  l7 = newl; Line(l7) = {p7, p8};
  // l8 = newl; Line(l8) = {p8, p5};

  // l9 = newl; Line(l9) = {p2, p5};

  pList[0] = p3; // First point label
  nPoints = 21; // Number of discretization points (top-right point of the inlet region)
  For i In {1 : nPoints}
    x = Xi + L/2*i/(nPoints + 1);
    pList[i] = newp;
    Point(pList[i]) = {x,
                  (R_base * (1 - f0/2 *(1 + Cos(2.0*Pi * (x-x0)/L))) - dR[k]),
                  0,
                  h};
  EndFor
  pList[nPoints+1] = p9; // Last point label (top-left point of the outlet region)
  pList2[0] = p9;
  For i In {1 : nPoints}
    x = Xi + L/2*(1 + i/(nPoints + 1));
    pList2[i] = newp;
    Point(pList2[i]) = {x,
                  (R_base * (1 - f0/2 *(1 + Cos(2.0*Pi * (x-x0)/L))) - dR[k]),
                  0,
                  h};
  EndFor
  pList2[nPoints+1] = p8; // Last point label (top-left point of the outlet region)
  l10 = newl; Spline(l10) = pList[];
  l20 = newl; Spline(l20) = pList2[];
  // dum = newp; Point(dum) = {Xi + L/2, R_base*(1-f0) - dR[k], 0, h};

  // Transfinite Line {9, 10} = Ceil(L/h) Using Progression 1;
  // Transfinite Line {4, -2, 8, -6} = Ceil(R/h) Using Progression 1.1;
  // Transfinite Line {1, 3} = Ceil(Xi/h) Using Progression 1;
  // Transfinite Line {5, 7} = Ceil(Xo/h) Using Progression 1;

  // l11 = newl; Line Loop(l11) = {l4, l1, l2, l3};
  // l12 = newl; Line Loop(l12) = {l2, l10, l20, l8, -l9};
  // l13 = newl; Line Loop(l13) = {l8, l5, l6, l7};
  // s1 = news; Plane Surface(s1) = {l11};
  // s2 = news; Plane Surface(s2) = {l12};
  // s3 = news; Plane Surface(s3) = {l13};

  l_loop = newl; Line Loop(l_loop) = {l0, l6, l7, -l20, -l10, l3, l4};
  s_loop = news; Plane Surface(s_loop) = {l_loop};


  /*
  // Local refinement
  Field[1] = Attractor;
  Field[1].NodesList = {p9};  // middle point
  // alternatively
  // Field[1].EdgesList = {l10, l20};
  // Field[1].NNodesByEdge = 100;

  Field[2] = Threshold;
  Field[2].IField = 1;
  Field[2].LcMin = h/3.;
  Field[2].LcMax = h;
  Field[2].DistMin = 0.025;
  Field[2].DistMax = 0.1;
  */

  // Background Field = 2;

  // Boundary Layer
  fname_k = Sprintf(fname);
  If (boundary_layer == 1)
    Line Loop(100) = {l10, l20};

    Field[3] = BoundaryLayer;
    Field[3].EdgesList = {100};
    // Field[3].NodesList = {3, 20};
    // Field[3].NodesList = {3, 9, 10:32};
    Field[3].NodesList = {p9, 24:33};
    Field[3].hfar = h;
    Field[3].hwall_n = h/6.; // 0.00000625;
    Field[3].hwall_t = h/6.; // 0.0000015;
    Field[3].thickness = 0.25;
    Field[3].ratio = 1.05;
    // Field[3].AnisoMax = 1;
    Field[3].Quads = 0;
    // Field[3].IntersectMetrics = 1;

    BoundaryLayer Field = 3;

    fname_k = StrCat(fname, "bl_");

  EndIf


  // Physical regions


  Physical Line(1) = {l4};   // inlet
  Physical Line(2) = {l6};   // outlet
  // Physical Line(3) = {l1, l9, l5};   // symmetry
  Physical Line(3) = {l0};   // symmetry
  // Physical Line(4) = {l3, l10, l20, l7};   // wall

  If (Nseg == 1)
  Physical Line(4) = {l3, l10, l20, l7};   // navier-slip
  EndIf
  If (Nseg == 2)
    Physical Line(4) = {l3, l7};   // NS, straight
    Physical Line(5) = {l10, l20};   // NS, cosine
  EndIf
  If (Nseg == 3)
    Physical Line(4) = {l3, l7};   // NS, straight
    Physical Line(5) = {l10};   // NS, cosine 1
    Physical Line(6) = {l20};   // NS, cosine 2
  EndIf
  If (Nseg == 4)
    Physical Line(4) = {l3};   // NS, straight 1
    Physical Line(5) = {l7};   // NS, straight 2
    Physical Line(6) = {l10};   // NS, cosine 1
    Physical Line(7) = {l20};   // NS, cosine 2
  EndIf

  // Physical Surface(0) = {s1, s2, s3};
  Physical Surface(0) = {s_loop};

  // tmpstr = Sprintf("Lc%g_L%g_f%g_d%g_ns%g_h%g.msh", L, len, f0, dR[k], Nseg, h);

  For i In {0 : reflevels }
    If (i == 0)
      Mesh 2;
    EndIf
    If (i > 0)
      RefineMesh;
    EndIf
    // Save Sprintf("coarc2d_bl_sc_f%g_d%g_ns%g_h%g.msh", f0, dR[k], Nseg, h/(2*i));
    tmpstr = Sprintf("Lc%g_L%g_f%1.1f_d%g_ns%g_h%g.msh", L, len, f0, dR[k], Nseg, h/(2^i));
    Save StrCat(fname_k, tmpstr);
  EndFor
EndFor

