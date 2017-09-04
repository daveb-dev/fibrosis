// 2D pipe

h = 0.1;
reflevels = 2;
L = 5;

Mesh.CharacteristicLengthFromPoints = 1;

Point(1) = {0, 0, 0, h};
Point(2) = {L, 0, 0, h};
Point(3) = {L, 1, 0, h};
Point(4) = {0, 1, 0, h};

Line(1) = {1, 2}; // bottom
Line(2) = {2, 3}; // right
Line(3) = {3, 4}; // -top
Line(4) = {4, 1}; // left

Line Loop(5) = {1, 2, 3, 4};

Plane Surface(1) = {5};

Physical Line(1) = {4};   // inlet
Physical Line(2) = {2};   // outlet
Physical Line(3) = {3};   // top
Physical Line(4) = {1};   // bottom


Physical Surface(0) = {1};

Mesh 2;
Save Sprintf("pipe2d_h%g.msh", h);
For i In {1 : reflevels }
  RefineMesh;
  Save Sprintf("pipe2d_h%g.msh", h/(2^i));
EndFor
