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
R = 1.0;  // cm
Xi = 1.0;
Xo = 2.0;
L = 2.0;

x0 = Xi + L/2.0;
f0 = 0.2; // [0;1]

angle = Pi/2.0;


/*
  h = 5;
 
  Xi = 100; // um
  Xo = 100; // um
  L = 100.0; // um
 
  R = 50.0;   // um
  f0 = 0.5;   // 0--1
 
  Z = 5;
*/

Point(1) = {0, 0, 0, h};
Point(2) = {Xi, 0, 0, h};
Point(3) = {Xi, R, 0, h};
Point(4) = {0, R, 0, h};

Point(5) = {Xi + L, 0, 0, h};
Point(6) = {Xi + L + Xo, 0, 0, h};
Point(7) = {Xi + L + Xo, R, 0, h};
Point(8) = {Xi + L, R, 0, h};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Line(9) = {2, 5};

pList[0] = 3; // First point label
nPoints = 21; // Number of discretization points (top-right point of the inlet region)
For i In {1 : nPoints}
  x = Xi + L*i/(nPoints + 1);
  pList[i] = newp;
  Point(pList[i]) = {x,
                ( R * (1 - f0/2 *(1 + Cos(2.0*Pi * (x-x0)/L) ) )),
                0,
                h};
EndFor
pList[nPoints+1] = 8; // Last point label (top-left point of the outlet region)
Spline(newl) = pList[];

// Transfinite Line {9, 10} = Ceil(L/h) Using Progression 1;
// Transfinite Line {4, -2, 8, -6} = Ceil(R/h) Using Progression 1.1;
// Transfinite Line {1, 3} = Ceil(Xi/h) Using Progression 1;
// Transfinite Line {5, 7} = Ceil(Xo/h) Using Progression 1;

Line Loop(11) = {4, 1, 2, 3};
Plane Surface(12) = {11};
Line Loop(13) = {2, 10, 8, -9};
Plane Surface(14) = {13};
Line Loop(15) = {8, 5, 6, 7};
Plane Surface(16) = {15};
// Transfinite Surface {14,12,16};
// Recombine Surface {14,12,16};

u[] = Extrude {{1, 0, 0}, {0, 0, 0}, 2*Pi/3} { Surface{12, 14, 16}; };
v[] = Extrude {{1, 0, 0}, {0, 0, 0}, 2*Pi/3} { Surface{u[0], u[5], u[10]}; };
w[] = Extrude {{1, 0, 0}, {0, 0, 0}, 2*Pi/3} { Surface{v[0], v[5], v[10]}; };
Printf("0    %g", u[0]);
Printf("1    %g", u[1]);
Printf("2    %g", u[2]);
Printf("3    %g", u[3]);
Printf("4    %g", u[4]);
Printf("5    %g", u[5]);
Printf("6    %g", u[6]);
Printf("6    %g", u[7]);
Printf("8    %g", u[8]);
Printf("9    %g", u[9]);
Printf("10    %g", u[10]);
Printf("11    %g", u[11]);
Printf("12    %g", u[12]);
Printf("13    %g", u[13]);
Printf("14    %g", u[14]);

Physical Surface(1) = {u[2], v[2], w[2]};   // Inlet
Physical Surface(2) = {u[13], v[13], w[13]};   // Outlet
Physical Surface(3) = {u[4], u[8], u[14], v[4], v[8], v[14], w[4], w[8], w[14]};   // Wall

Physical Volume(1) = {u[1], u[6], u[11], v[1], v[6], v[11], w[1], w[6], w[11]};


Mesh 3;
Save Sprintf('coarc3d_r0.msh');
RefineMesh;
Save Sprintf('coarc3d_r1.msh');
