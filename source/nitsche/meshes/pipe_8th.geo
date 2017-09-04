// Geometric parameters
Ri=1;
h=0.1;
Lx=4;

Mesh.CharacteristicLengthExtendFromBoundary = 1;
// Mesh.CharacteristicLengthFactor = 1;
Mesh.Algorithm3D = 1; //Frontal (4) Delaunay(1)
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

Mesh.CharacteristicLengthMin=0.99*h;
Mesh.CharacteristicLengthMax=1.01*h;

// Base circle   sin(45°) = cos(45°) = 0.7071067811
Point(1) = {0, 0, 0, h};
Point(2) = {0, Ri, 0, h};
Point(3) = {0, 0, -Ri, h};
Line(1) = {1, 2};
Circle(2) = {2, 1, 3};
Line(3) = {3, 1};
Line Loop(5) = {1, 2, 3};
Plane Surface(6) = {5};


// Extrusion
ex = Extrude {Lx, 0, 0} {
  Surface{6};
};

Printf("%g", ex[0]);
Printf("%g", ex[1]);
Printf("%g", ex[2]);
Printf("%g", ex[3]);
Printf("%g", ex[4]);

Physical Surface(1) = {6};  // inlet
Physical Surface(2) = {23};   // outlet
Physical Surface(3) = {18};  // wall
Physical Surface(4) = {14, 22};  // symmetry

Physical Volume(0) = ex[1];

// Mesh
/* Mesh.SaveAll=0; */
/* Mesh.Format=1; */

Mesh 3;
Save Sprintf("pipe4_r0.msh");
RefineMesh;
Save Sprintf("pipe4_r1.msh");

