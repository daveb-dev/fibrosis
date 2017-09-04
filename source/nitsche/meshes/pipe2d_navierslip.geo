// 2D pipe

Nseg = 2;
h = 0.05;
reflevels = 0;
L = 6;

Mesh.CharacteristicLengthFromPoints = 1;

i = -1;
// dR[i++] = 0;  // reference mesh
dR[i++] = 0.1;
// dR[i++] = 0.05;

lastInd = i;

For k In {0:lastInd}
  // Delete physical groups
  Delete Model;

  R = 1 - dR[k];

  Point(1) = {0, 0, 0, h};
  Point(2) = {L, 0, 0, h};
  Point(3) = {L, R, 0, h};
  Point(4) = {0, R, 0, h};

  Point(5) = {0.25*L, R, 0, h};
  Point(6) = {0.5*L, R, 0, h};
  Point(7) = {0.75*L, R, 0, h};

  Line(1) = {1, 2}; // bottom
  Line(2) = {2, 3}; // right
  // Line(3) = {3, 4}; // -top
  Line(3) = {3, 7};
  Line(4) = {7, 6};
  Line(5) = {6, 5};
  Line(6) = {5, 4};
  Line(7) = {4, 1}; // left

  Line Loop(8) = {1, 2, 3, 4, 5, 6, 7};

  Plane Surface(1) = {8};

  Physical Line(1) = {7};   // inlet
  Physical Line(2) = {2};   // outlet
  Physical Line(3) = {1};   // bottom

  If (Nseg == 1)
    Physical Line(4) = {3, 4, 5, 6};
  EndIf
  If (Nseg == 2)
    Physical Line(4) = {3, 4};
    Physical Line(5) = {5, 6};
  EndIf
  If (Nseg == 4)
    Physical Line(4) = {6};
    Physical Line(5) = {5};
    Physical Line(6) = {4};
    Physical Line(7) = {3};
  EndIf


  Physical Surface(0) = {1};

  Mesh 2;
  Save Sprintf("pipe2d_d%g_ns%g_h%g.msh", dR[k], Nseg, h);
  For i In {1 : reflevels }
    RefineMesh;
    Save Sprintf("pipe2d_d%g_ns%g_h%g.msh", dR[k], Nseg, h/(2^i));
  EndFor
EndFor
