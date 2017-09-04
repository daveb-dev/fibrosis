// Backward facing step, 2D
// Gresho 1993, etc; Biswas 2004 doi 0.1115/1.1760532

h = 0.5;
s = 0.5;
H = h + s;
Lu = 20*h;
Ld = 20*h;

hc = 0.1;
hcp = hc;
reflevels = 2;

Mesh.CharacteristicLengthFromPoints = 1;

Point(1) = {-Lu, h, 0, hc};
Point(2) = {-Lu, 0, 0, hc};
Point(3) = {0, 0, 0, hcp};
Point(4) = {0, -s, 0, hc};
Point(5) = {Ld, -s, 0, hc};
Point(6) = {Ld, h, 0, hc};

Line(1) = {1, 2};   // inlet
Line(2) = {2, 3};   // upstream channel bottom
Line(3) = {3, 4};   // step vertical boundary
Line(4) = {4, 5};   // downstream channel bottom
Line(5) = {5, 6};   // outlet
Line(6) = {6, 1};   // channel top

Line Loop(7) = {1, 2, 3, 4, 5, 6};

Plane Surface(1) = {7};

Physical Line(1) = {1};   // inlet
Physical Line(2) = {5};   // outlet
Physical Line(3) = {2, 3, 4, 6};   // walls

Physical Surface(0) = {1};

Mesh 2;
Save Sprintf("bfs2d_h%g.msh", hc);
For i In {1 : reflevels }
  RefineMesh;
  Save Sprintf("bfs2d_h%g.msh", hc/(2^i));
EndFor
