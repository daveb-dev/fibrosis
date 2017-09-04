// Geometric parameters
Ri=1;
h=0.12;
Lz=5;
Mesh.CharacteristicLengthExtendFromBoundary = 1;
// Mesh.CharacteristicLengthFactor = 1;
Mesh.Algorithm3D = 4; //Frontal (4) Delaunay(1)
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

Mesh.CharacteristicLengthMin=0.99*h;
Mesh.CharacteristicLengthMax=1.01*h;

// Base circle
Point(1) = {0, 0, 0, h};
Point(2) = {-Ri, 0, 0, h};
Point(3) = {Ri, 0, 0, h};
Point(4) = {0, 0, Ri, h};
Point(5) = {0, 0, -Ri, h};
Circle(1) = {2, 1, 5};
Circle(2) = {5, 1, 3};
Circle(3) = {3, 1, 4};
Circle(4) = {4, 1, 2};
Line Loop(5) = {1, 2, 3,4};
Plane Surface(6) = {5};


// Extrusion
pipe_vol = Extrude {0, Lz, 0} {Surface{6};};

// physical entities

Physical Surface(0) = {28};  								// inlet face
Physical Surface(1) = {6};								// outlet surface
Physical Surface(2) = {27, 15, 23, 19};  						// Manto
Physical Volume(0) = {pipe_vol[1]};

