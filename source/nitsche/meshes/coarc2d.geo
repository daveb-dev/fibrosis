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

h = 0.1;
h2 = 0.1;
reflevels = 2;
// Define mesh alghoritm and optimize process
// Mesh.Algorithm3D 				= 4;
// Mesh.Optimize 					= 1;
// Mesh.OptimizeNetgen 		= 1;

Mesh.CharacteristicLengthFromPoints = 1;
// Bounds for lengths of elements
// Mesh.CharacteristicLengthMin			= 0.99*h;
// Mesh.CharacteristicLengthMax			= 1.01*h;

R_base = 1.0;  // cm
Xi = 1.0;
Xo = 10.0;
L = 2.0;

x0 = Xi + L/2.0;
f0 = 0.6; // [0;1]

i = -1;
dR[i++] = 0;  // reference mesh
// dR[i++] = 0.1;
// dR[i++] = 0.15;
// dR[i++] = 0.1;
lastInd = i;

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
  p2 = newp; Point(p2) = {Xi, 0, 0, h};
  p3 = newp; Point(p3) = {Xi, R, 0, h2};
  p4 = newp; Point(p4) = {0, R, 0, h};

  p5 = newp; Point(p5) = {Xi + L, 0, 0, h};
  p6 = newp; Point(p6) = {Xi + L + Xo, 0, 0, h};
  p7 = newp; Point(p7) = {Xi + L + Xo, R, 0, h};
  p8 = newp; Point(p8) = {Xi + L, R, 0, h2};

  l1 = newl; Line(l1) = {p1, p2};
  l2 = newl; Line(l2) = {p2, p3};
  l3 = newl; Line(l3) = {p3, p4};
  l4 = newl; Line(l4) = {p4, p1};

  l5 = newl; Line(l5) = {p5, p6};
  l6 = newl; Line(l6) = {p6, p7};
  l7 = newl; Line(l7) = {p7, p8};
  l8 = newl; Line(l8) = {p8, p5};

  l9 = newl; Line(l9) = {p2, p5};

  pList[0] = 3; // First point label
  nPoints = 21; // Number of discretization points (top-right point of the inlet region)
  For i In {1 : nPoints}
    x = Xi + L*i/(nPoints + 1);
    pList[i] = newp;
    Point(pList[i]) = {x,
                  (R_base * (1 - f0/2 *(1 + Cos(2.0*Pi * (x-x0)/L))) - dR[k]),
                  0,
                  h2};
  EndFor
  pList[nPoints+1] = 8; // Last point label (top-left point of the outlet region)
  l10 = newl; Spline(l10) = pList[];
  // dum = newp; Point(dum) = {Xi + L/2, R_base*(1-f0) - dR[k], 0, h2};

  // Transfinite Line {9, 10} = Ceil(L/h) Using Progression 1;
  // Transfinite Line {4, -2, 8, -6} = Ceil(R/h) Using Progression 1.1;
  // Transfinite Line {1, 3} = Ceil(Xi/h) Using Progression 1;
  // Transfinite Line {5, 7} = Ceil(Xo/h) Using Progression 1;

  l11 = newl; Line Loop(l11) = {l4, l1, l2, l3};
  l12 = newl; Line Loop(l12) = {l2, l10, l8, -l9};
  l13 = newl; Line Loop(l13) = {l8, l5, l6, l7};
  s1 = news; Plane Surface(s1) = {l11};
  s2 = news; Plane Surface(s2) = {l12};
  s3 = news; Plane Surface(s3) = {l13};
  // Transfinite Surface {14,12,16};
  // Recombine Surface {14,12,16};

  // v[] = Extrude {{1, 0, 0}, {0, 0, 0}, angle} { Surface{12, 14, 16}; };

  Physical Line(1) = {l4};   // inlet
  Physical Line(2) = {l6};   // outlet
  Physical Line(3) = {l3, l10, l7};   // wall
  Physical Line(4) = {l1, l9, l5};   // symmetry

  Physical Surface(0) = {s1, s2, s3};

  Mesh 2;
  Save Sprintf("coarc2dl_f%g_d%g_h%g.msh", f0, dR[k], h);
  For i In {1 : reflevels }
    RefineMesh;
    Save Sprintf("coarc2dl_f%g_d%g_h%g.msh", f0, dR[k], h/(2^i));
  EndFor
EndFor
