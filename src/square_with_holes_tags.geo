ls = 0.1;
//количество отверстий вдоль оси Ox
n = 5;
//количество отверстий вдоль оси Oy
m = 5;
// радиус окружностей
r = 0.2;
//размер патча вдоль оси Ox
a = 10;
//размер патча вдоль оси Oy
b = 10;
//толщина образца
th = 0.1;

Point(1) = {0, 0, 0, ls};
Point(2) = {a, 0, 0, ls};
Point(3) = {a, b, 0, ls};
Point(4) = {0, b, 0, ls};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(1) = {1, 2, 3, 4};

X = 0;
Y = 0;
loops[] = {};
points[] = {};
points_down[] = {};
points_up[] = {};
points_left[] = {};
points_right[] = {};

Function CircleLoop
    pp = newp;
    Point(pp) = {X, Y, 0, ls};
    Point(pp+1) = {X+r, Y, 0, ls};
    Point(pp+2) = {X, Y+r, 0, ls};
    Point(pp+3) = {X-r, Y, 0, ls};
    Point(pp+4) = {X, Y-r, 0, ls};
    points[] = {};
    
    //points[] += pp;
    If (position == 4)
        points[] += pp + 1;
    EndIf

    If (position == 2)
        points[] += pp + 2;
    EndIf

    If (position == 3)
        points[] += pp + 3;
    EndIf

    If (position == 1)
        points[] += pp + 4;
    EndIf

    //points[] += pp + 2;
    //points[] += pp + 3;
    //points[] += pp + 4;
    //pp = new
    //Physical Point(pp) = {pp + 1};

    lp = newl;
    Circle(lp) = {pp+1, pp, pp + 2};
    Circle(lp+1) = {pp+2, pp, pp + 3};
    Circle(lp+2) = {pp+3, pp, pp + 4};
    Circle(lp+3) = {pp+4, pp, pp + 1};

    llp = newll;
    Line Loop(llp) = {lp, lp+1, lp+2, lp+3};
    loops[] += llp;

    //Physical Curve(lp) = {pp, pp +3};

Return

dist_to_border = 1.15;
//da = 2 * sqrt(2) * r + 0.4;
da = 0.9;
If (da < 2 * Sqrt(2) * r)
    da = 2 * Sqrt(2) * r + 0.00001;
EndIf
circle2cirlce = (a - (dist_to_border + da) * 2) / (n - 1);

//circle2cirlce = 1.6;
//circle_coords[] = {1.7, 3.4, 5.1, 6.8, 8.5};
//circle_coords[] = {1.7, 2.6, 3.5, 4.4, 5.3};

For i In {0:4}

    position = 1;
    X = dist_to_border + da + circle2cirlce * i;
    Y = dist_to_border;
    Call CircleLoop;
    points_down[] += points[];

    position = 2;
    X = dist_to_border + da + circle2cirlce * i;
    Y = b - dist_to_border;
    Call CircleLoop;
    points_up[] += points[];

    position = 3;
    X = dist_to_border;
    Y = da + dist_to_border + circle2cirlce * i;
    Call CircleLoop;
    points_left[] += points[];

    position = 4;
    X = a - dist_to_border;
    Y = da + dist_to_border  + circle2cirlce * i;
    Call CircleLoop;
    points_right[] += points[];
    
EndFor
Physical Point(131) = {points_down[]};
Physical Point(132) = {points_up[]};
Physical Point(141) = {points_left[]};
Physical Point(142) = {points_right[]};

Plane Surface(1) = {1, loops[]};
Physical Surface(2) = {1};
//Extrude{0,0,1}{Surface{1};}

//MeshSize{PointsOf{Volume{:};}} = 0.1;

//Mesh.CharacteristicLengthMin = 0.8;
//Mesh.CharacteristicLengthMax = 0.4;
Mesh 2;
//+
//Transfinite Surface {1};
//+
//Recombine Surface {1};
//Mesh.Format = 16;
Save "rake.vtk";
