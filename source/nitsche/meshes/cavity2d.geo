// 2D driven cavity mesh
// Top boundary: driven lid
// Left, right, bottom boundaries: non-moving walls


h = 0.1;
reflevels = 3;

Mesh.CharacteristicLengthFromPoints = 1;

Point(1) = {0, 0, 0, h};
Point(2) = {1, 0, 0, h};
Point(3) = {1, 1, 0, h};
Point(4) = {0, 1, 0, h};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(5) = {1, 2, 3, 4};

Plane Surface(1) = {5};

Physical Line(1) = {3};   // lid
Physical Line(2) = {1, 2, 4};   // wall

Physical Surface(0) = {1};

Mesh 2;
Save Sprintf("cavity2d_h%g.msh", h);
For i In {1 : reflevels }
  RefineMesh;
  Save Sprintf("cavity2d_h%g.msh", h/(2^i));
EndFor
