/*
 * 
 * 	Mesh generator for blood vessels with diferrent coarctations grades
 * 	accordingly to the proportion of r and rc
 * 
 * 	FG - 2016
 * 
 */

h  	= 0.3;		// element size
r  	= 1.0;		// normal vessel-radius
rc 	= 0.4;		// coarted vessel radius
L1	= 1.8 ;		// half of the coarted length
L2	= 2.5 - L1;	// half of healthy vessel

aux	= L2/3;		// parameter to fix control points of splines (use aux = 0 to create a pure cylinder)
//aux = 0.0;

// Define mesh alghoritm and optimize process
Mesh.Algorithm3D 				= 4;
Mesh.Optimize 					= 1;
Mesh.OptimizeNetgen 				= 1;

// Bounds for lengths of elements
Mesh.CharacteristicLengthMin			= 0.99*h;
Mesh.CharacteristicLengthMax			= 1.01*h;

// ****************************
//  Elementary Elements

// Points to create splines and arc of circles
p0 = newp; Point(p0) = {-r , L1       , 0, h};
p1 = newp; Point(p1) = {-rc, L1 + L2  , 0, h};
p2 = newp; Point(p2) = {-r , L1 + 2*L2, 0, h};

p3 = newp; Point(p3) = { r , L1       , 0, h};
p4 = newp; Point(p4) = { rc, L1 + L2  , 0, h};
p5 = newp; Point(p5) = { r , L1 + 2*L2, 0, h};

p6 = newp; Point(p6) = { 0 , L1       , r , h};
p7 = newp; Point(p7) = { 0 , L1 + L2  , rc, h};
p8 = newp; Point(p8) = { 0 , L1 + 2*L2, r , h};

p9  = newp; Point(p9 ) = { 0 , L1       , -r , h};
p10 = newp; Point(p10) = { 0 , L1 + L2  , -rc, h};
p11 = newp; Point(p11) = { 0 , L1 + 2*L2, -r , h};

// Center of circles
p12 = newp; Point(p12) = { 0 , L1        , 0 , h};
p13 = newp; Point(p13) = { 0 , L1 + 2*L2 , 0 , h};
p14 = newp; Point(p14) = { 0 , L1 + L2, 0, h};

// Auxiliar spline control points
a1  = newp; Point(a1) = {-r +  aux/2, L1 + aux     , 0, h};
a2  = newp; Point(a2) = {-rc - aux/2, L1 + L2 - aux, 0, h};
a3  = newp; Point(a3) = {-rc - aux/2, L1 + L2 + aux, 0, h};
a4  = newp; Point(a4) = {-r +  aux/2, L1 + 2*L2 - aux     , 0, h};

a5  = newp; Point(a5) = { 0 , L1 + aux       , r  - aux/2,  h};
a6  = newp; Point(a6) = { 0 , L1 + L2 - aux  , rc + aux/2 , h};
a7  = newp; Point(a7) = { 0 , L1 + L2 + aux  , rc + aux/2 , h};
a8  = newp; Point(a8) = { 0 , L1 + 2*L2 - aux, r  - aux/2 , h};

a9  = newp; Point(a9)  = {r  - aux/2, L1 + aux     , 0, h};
a10 = newp; Point(a10) = {rc + aux/2, L1 + L2 - aux, 0, h};
a11 = newp; Point(a11) = {rc + aux/2, L1 + L2 + aux, 0, h};
a12 = newp; Point(a12) = {r  - aux/2, L1 + 2*L2 - aux     , 0, h};

a13  = newp; Point(a13) = { 0 , L1 + aux       , -r + aux/2,  h};
a14  = newp; Point(a14) = { 0 , L1 + L2 - aux  , -rc - aux/2 , h};
a15  = newp; Point(a15) = { 0 , L1 + L2 + aux  , -rc - aux/2 , h};
a16  = newp; Point(a16) = { 0 , L1 + 2*L2 - aux, -r  + aux/2 , h};

// Contraction splines
sp1 = newl; Spline(sp1) = {p0, a1, a2, p1};
sp2 = newl; Spline(sp2) = {p1, a3, a4, p2};
sp3 = newl; Spline(sp3) = {p6, a5, a6, p7};
sp4 = newl; Spline(sp4) = {p7, a7, a8, p8};
sp5 = newl; Spline(sp5) = {p3, a9, a10, p4};
sp6 = newl; Spline(sp6) = {p4, a11,a12, p5};
sp7 = newl; Spline(sp7) = {p9, a13,a14, p10};
sp8 = newl; Spline(sp8) = {p10,a15,a16, p11};

// Down Contraction Circle
c1 = newreg; Circle(c1) = {p6, p12, p0};
c2 = newreg; Circle(c2) = {p0, p12, p9};
c3 = newreg; Circle(c3) = {p9, p12, p3};
c4 = newreg; Circle(c4) = {p3, p12, p6};

// Mid Contraction Circle
c9  = newreg; Circle(c9)  = {p1,  p14, p10};
c10 = newreg; Circle(c10) = {p10, p14, p4};
c11 = newreg; Circle(c11) = {p4,  p14, p7};
c12 = newreg; Circle(c12) = {p7,  p14, p1};

// Up Contraction Circle
c5 = newreg; Circle(c5) = {p2, p13, p8};
c6 = newreg; Circle(c6) = {p8, p13, p5};
c7 = newreg; Circle(c7) = {p5, p13, p11};
c8 = newreg; Circle(c8) = {p11, p13, p2};

// Contraction Surfaces
surf1_contour = newl; Line Loop(surf1_contour) = {sp1,-c12, -sp3,   c1};
surf2_contour = newl; Line Loop(surf2_contour) = {sp2,  c5, -sp4,  c12};
surf3_contour = newl; Line Loop(surf3_contour) = {sp5, c11, -sp3,  -c4};
surf4_contour = newl; Line Loop(surf4_contour) = {sp6, -c6, -sp4, -c11};
surf5_contour = newl; Line Loop(surf5_contour) = {c3,  sp5, -c10, -sp7};
surf6_contour = newl; Line Loop(surf6_contour) = {sp6,  c7, -sp8,  c10};
surf7_contour = newl; Line Loop(surf7_contour) = {sp1,  c9, -sp7,  -c2};
surf8_contour = newl; Line Loop(surf8_contour) = {sp2, -c8, -sp8,  -c9};

surf1 	      = news; Ruled Surface(surf1)     = {surf1_contour};
surf2 	      = news; Ruled Surface(surf2)     = {surf2_contour};
surf3 	      = news; Ruled Surface(surf3)     = {surf3_contour};
surf4 	      = news; Ruled Surface(surf4)     = {surf4_contour};
surf5 	      = news; Ruled Surface(surf5)     = {surf5_contour};
surf6 	      = news; Ruled Surface(surf6)     = {surf6_contour};
surf7 	      = news; Ruled Surface(surf7)     = {surf7_contour};
surf8 	      = news; Ruled Surface(surf8)     = {surf8_contour};


// Contraction Volume
lcircle_up   = newl; Line Loop(lcircle_up)   = { c5, c6, c7, c8 };
lcircle_down = newl; Line Loop(lcircle_down) = { c1, c2, c3, c4 };
circle_up    = news; Plane Surface(circle_up)   = {lcircle_up};
circle_down  = news; Plane Surface(circle_down) = {lcircle_down};
manto        = news; Surface Loop(manto)    = {surf1, surf2, surf3, surf4, surf5, surf6, surf7, surf8, circle_up, circle_down};
coarted_part = newv; Volume (coarted_part)  = {manto};

// The rest of the domain
pipe_up   = Extrude {0,  L1, 0} { Surface{circle_up  }; };
pipe_down = Extrude {0, -L1, 0} { Surface{circle_down}; };


// ****************************
//  Physical Elements

Physical Surface(0) = {64};  								// inlet face
Physical Surface(1) = {86};								// outlet surface
Physical Surface(2) = {surf1, surf2, surf3, surf4, surf5, surf6, surf7, surf8, 63, 59, 55, 51, 85, 81, 77, 73};  	// Manto

Physical Volume(0) = {pipe_up[1], pipe_down[1], coarted_part};