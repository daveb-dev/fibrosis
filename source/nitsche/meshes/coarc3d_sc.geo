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

Nseg = 1;   // 1, 2, 3, 4

h = 0.05;
h2 = 0.05;
reflevels = 0;
// Define mesh alghoritm and optimize process
// Mesh.Algorithm3D 				= 4;
Mesh.Optimize 					= 1;
Mesh.OptimizeNetgen 		= 1;

Mesh.CharacteristicLengthFromPoints = 1;
// Bounds for lengths of elements
// Mesh.CharacteristicLengthMin			= 0.99*h;
// Mesh.CharacteristicLengthMax			= 1.01*h;

// total length: long: 10., normal: 5.

R_base = 1.0;  // cm
Xi = 1.0;
Xo = 3.;
L = 1.;      // short: 1.0, sc2: 0.5, normal: 2.0

x0 = Xi + L/2.0;
f0 = 0.6; // [0;1]

i = -1;
dR[i++] = 0;  // reference mesh
// dR[i++] = 0.05;
// dR[i++] = 0.1;
// dR[i++] = 0.15;
// dR[i++] = 0.2;
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
  p3 = newp; Point(p3) = {Xi, R, 0, h};    // start of cosine
  p4 = newp; Point(p4) = {0, R, 0, h};

  p5 = newp; Point(p5) = {Xi + L, 0, 0, h};
  p6 = newp; Point(p6) = {Xi + L + Xo, 0, 0, h};
  p7 = newp; Point(p7) = {Xi + L + Xo, R, 0, h};
  p8 = newp; Point(p8) = {Xi + L, R, 0, h};   // end of cosine

  p9 = newp; Point(p9) = {Xi + L/2, R_base*(1.-f0) - dR[k], 0, h2};  // middle point

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
                  h2};
  EndFor
  pList[nPoints+1] = p9; // Last point label (top-left point of the outlet region)
  pList2[0] = p9;
  For i In {1 : nPoints}
    x = Xi + L/2*(1 + i/(nPoints + 1));
    pList2[i] = newp;
    Point(pList2[i]) = {x,
                  (R_base * (1 - f0/2 *(1 + Cos(2.0*Pi * (x-x0)/L))) - dR[k]),
                  0,
                  h2};
  EndFor
  pList2[nPoints+1] = p8; // Last point label (top-left point of the outlet region)
  l10 = newl; Spline(l10) = pList[];
  l20 = newl; Spline(l20) = pList2[];


  v[] = Extrude { {1, 0, 0}, {0, 0, 0}, 2./3.*Pi} { Line{l3, l4, l6, l7, l10,
  l20, l0}; };
  For itmp In {0: 21}
    Printf("%g  %g", itmp, v[itmp]);
  EndFor
  v2[] = Extrude { {1, 0, 0}, {0, 0, 0}, 2./3.*Pi} { Line{v[0], v[4], v[7],
    v[10], v[14], v[18]}; };
  v3[] = Extrude { {1, 0, 0}, {0, 0, 0}, 2./3.*Pi} { Line{v2[0], v2[4], v2[7],
    v2[10], v2[14], v2[18]}; };

  Physical Surface(1) = {v[5], v2[5], v3[5]};   // Inlet
  Physical Surface(2) = {v[8], v2[8], v3[8]};   // Outlet
  Physical Surface(3) = {v[1], v[15], v[19], v[11], v2[1], v2[15], v2[19],
    v2[11], v3[1], v3[15], v3[19], v3[11]};   // Wall

  s_loop = news; Surface Loop(s_loop) = {v[5], v2[5], v3[5], v[8], v2[8],
  v3[8], v[1], v[15], v[19], v[11], v2[1], v2[15], v2[19], v2[11], v3[1],
  v3[15], v3[19], v3[11]};

	Volume(1) = {s_loop};
	Physical Volume(1) = {1};


  // Physical Line(1) = {l4};   // inlet
  // Physical Line(2) = {l6};   // outlet
  // Physical Line(3) = {l1, l9, l5};   // symmetry

  // If (Nseg == 1)
  // Physical Line(4) = {l3, l10, l20, l7};   // navier-slip
  // EndIf
  /* If (Nseg == 2)
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
 */
  // Physical Surface(0) = {s1, s2, s3};

	Mesh 3;
	Save Sprintf("coarc3d_sc_f%g_d%g_ns%g_h%g.msh", f0, dR[k], Nseg, h);
	For i In {1 : reflevels }
		RefineMesh;
		Save Sprintf("coarc3d_sc_f%g_d%g_ns%g_h%g.msh", f0, dR[k], Nseg, h/(2*i));
	EndFor 
EndFor

