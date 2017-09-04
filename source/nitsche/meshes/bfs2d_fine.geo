// Backward facing step, 2D
// Gresho 1993, etc; Biswas 2004 doi 0.1115/1.1760532

// Refined with Treshold/Attractor method

h = 0.5;
s = 0.5;
H = h + s;
Lu = 2*h;
Ld = 20*h;

lc = 0.1;
lcp = lc;
reflevels = 1;

Mesh.CharacteristicLengthFromPoints = 1;

Point(1) = {-Lu, h, 0, lc};
Point(2) = {-Lu, 0, 0, lc};
Point(3) = {0, 0, 0, lcp};
Point(4) = {0, -s, 0, lc};
Point(5) = {Ld, -s, 0, lc};
Point(6) = {Ld, h, 0, lc};

Line(1) = {1, 2};   // inlet
Line(2) = {2, 3};   // upstream channel bottom
Line(3) = {3, 4};   // step vertical boundary
Line(4) = {4, 5};   // downstream channel bottom
Line(5) = {5, 6};   // outlet
Line(6) = {6, 1};   // channel top

Line Loop(7) = {1, 2, 3, 4, 5, 6};

Plane Surface(1) = {7};


// Refinement
preflen = 4*h;
Point(7) = {preflen, 0, 0, lcp};
Line(8) = {3, 7};
Field[1] = Attractor;
// Field[1].NodesList = {3};  // corner point
Field[1].EdgesList = {8};  // corner point
Field[1].NNodesByEdge = 100;

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc/3.;
Field[2].LcMax = lc;
Field[2].DistMin = 0.05;
Field[2].DistMax = 0.4;

// Field[1] = Box;
// Field[1].VIn = lc / 4;
// Field[1].VOut = lc;
// Field[1].XMin = 0.0;
// Field[1].XMax = 3.0;
// Field[1].YMin = -0.2;
// Field[1].YMax = 0.1;

Background Field = 2;




Physical Line(1) = {1};   // inlet
Physical Line(2) = {5};   // outlet
Physical Line(3) = {2, 3, 4, 6};   // walls

Physical Surface(0) = {1};

Mesh 2;
Save Sprintf("bfs2df_h%g.msh", lc);
For i In {1 : reflevels }
  RefineMesh;
  Save Sprintf("bfs2df_h%g.msh", lc/(2^i));
EndFor
