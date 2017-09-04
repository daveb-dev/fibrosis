L = 1; /* voxel length */
dx1 = 0.05;
R = 0.398942280;  /* radius for a circle taking half of the unit square
area */
dx2 = dx1;

Point(1) = {-0.5*L, -0.5*L, -0.5*L, dx1};
Point(2) = { 0.5*L, -0.5*L, -0.5*L, dx1};
Point(3) = { 0.5*L, -0.5*L,  0.5*L, dx1};
Point(4) = {-0.5*L, -0.5*L,  0.5*L, dx1};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

lvox = newl;
Line Loop(lvox) = {1, 2, 3, 4};

p0 = newp; Point(p0) = {0, -0.5*L, 0, dx2};
p1 = newp; Point(p1) = {R, -0.5*L, 0, dx2};
p2 = newp; Point(p2) = {0, -0.5*L, R, dx2};
p3 = newp; Point(p3) = {-R, -0.5*L, 0, dx2};
p4 = newp; Point(p4) = {0, -0.5*L, -R, dx2};

c1 = newreg; Circle(c1) = {p1, p0, p2};
c2 = newreg; Circle(c2) = {p2, p0, p3};
c3 = newreg; Circle(c3) = {p3, p0, p4};
c4 = newreg; Circle(c4) = {p4, p0, p1};

lcirc = newl;
Line Loop(lcirc) = {c1, c2, c3, c4};

sc = news;
Plane Surface(sc) = {lcirc};
sv = news;
Plane Surface(sv) = {lvox, lcirc};

mat = Extrude {0, 1*L, 0} { Surface{sv}; };
pipe = Extrude {0, 1*L, 0} { Surface{sc}; };
/*
ex[0]: new created "top" surface of extrusion (y=0.5) (back matrix)
ex[1]: bottom Line (1)
ex[2]: 1st line in LineLoop (lvox): bottom (z=-0.5)
ex[3]: right (x=0.5)
ex[4]: top (z=0.5)
ex[5]: left (x=-0.5)
ex[6-9]: lcirc LineLoop (quarter cylinder mantles)
ex[10]: circle outlet
*/

/* Physical Surface */
Printf("EXTRUSION Srf Square\Circle");
Printf("back matrix: %g", mat[0]);
Printf("Volume(1):  %g", mat[1]);
Printf("bottom:  %g", mat[2]);
Printf("right:  %g", mat[3]);
Printf("top:  %g", mat[4]);
Printf("left:  %g", mat[5]);
Printf("interface:  %g", mat[6]);
Printf("interface:  %g", mat[7]);
Printf("interface:  %g", mat[8]);
Printf("interface:  %g", mat[9]);


Printf("EXTRUSION Circle");
Printf("back outlet: %g", pipe[0]);
Printf("Volume(2):  %g", pipe[1]);
Printf("interface:  %g", pipe[2]);
Printf("interface:  %g", pipe[3]);
Printf("interface:  %g", pipe[4]);
Printf("interface:  %g", pipe[5]);


/* // number from 0 to N for FEniCS compatibility */
Physical Surface(0) = {sv};  // front matrix {Sq/Cir}
Physical Surface(1) = {sc};  // front inlet
Physical Surface(2) = {mat[0]};  // back matrix {Sq/Cir}
Physical Surface(3) = {pipe[0]};  // back outlet
Physical Surface(4) = {mat[2]};  // bottom
Physical Surface(5) = {mat[3]};  // right
Physical Surface(6) = {mat[4]};  // top
Physical Surface(7) = {mat[5]};  // left
Physical Surface(8) = {mat[6], mat[7], mat[8], mat[9]};  // interface

Physical Volume(0) = {mat[1]};
Physical Volume(1) = {pipe[1]};


// MESHING
// run with: " gmsh vx.geo - "
Mesh.OptimizeNetgen = 1;
Mesh 3;
//SetOrder 2;
Save Sprintf("vx_r%02g.msh", 0);
For i In {1:3}
  RefineMesh;
//  SetOrder 2;
  Save Sprintf("vx_r%02g.msh", i);
EndFor
