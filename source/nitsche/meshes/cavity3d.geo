// 3D driven cavity mesh
// Top boundary: driven lid
// Left, right, bottom boundaries: non-moving walls


h = 0.1;
reflevels = 2;

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

ex = Extrude {0, 0, 1} { Surface{1}; };

Physical Surface(1) = {22};                   // lid
Physical Surface(2) = {18, 27, 26, 1, 14};    // wall

Physical Volume(0) = {1};

Mesh.OptimizeNetgen = 1;
// Mesh 3;
// Save Sprintf("cavity3d_h%g.msh", h);
// For i In {1 : reflevels }
//   RefineMesh;
//   Save Sprintf("cavity3d_h%g.msh", h/(2^i));
// EndFor
