#!/usr/bin/env wolframscript

validationDatapoints = {}
Do[
    x1o = {RandomReal[{-1, 1}], RandomReal[{-1, 1}]};
    x2o = {RandomReal[{-1, 1}], RandomReal[{-1, 1}]};
    x3o = {RandomReal[{-1, 1}], RandomReal[{-1, 1}]};

    v1o = {0,0};
    v2o = {0,0};
    v3o = {0,0};

    Tend = 80;

    m1 = RandomReal[{1,100}];
    m2 = RandomReal[{1,100}];
    m3 = RandomReal[{1,100}];

    G =  6.674*(10**-11);

    (*f=ma*)
    eqn1ax = m1 x1''[t] ==  ((m1 m2 (x1[t] - x2[t]))/((x1[t] - x2[t])^2 + (y1[t] - y2[t])^2)^(3/2)) - ((m1 m3 (x1[t] - x3[t]))/((x1[t] - x3[t])^2 + (y1[t] - y3[t])^2)^(3/2));
    eqn1ay = m1 y1''[t] ==  ((m1 m2 (y1[t] - y2[t]))/((x1[t] - x2[t])^2 + (y1[t] - y2[t])^2)^(3/2)) - ((m1 m3 (y1[t] - y3[t]))/((x1[t] - x3[t])^2 + (y1[t] - y3[t])^2)^(3/2));
    
    eqn2ax = m2 x2''[t] ==  ((m1 m2 (x1[t] - x2[t]))/((x1[t] - x2[t])^2 + (y1[t] - y2[t])^2)^(3/2)) - ((m2 m3 (x2[t] - x3[t]))/((x2[t] - x3[t])^2 + (y2[t] - y3[t])^2)^(3/2));
    eqn2ay = m2 y2''[t] ==  ((m1 m2 (y1[t] - y2[t]))/((x1[t] - x2[t])^2 + (y1[t] - y2[t])^2)^(3/2)) - ((m2 m3 (y2[t] - y3[t]))/((x2[t] - x3[t])^2 + (y2[t] - y3[t])^2)^(3/2));

    eqn3ax = m3 x3''[t] ==  ((m1 m3 (x1[t] - x3[t]))/((x1[t] - x3[t])^2 + (y1[t] - y3[t])^2)^(3/2)) + ((m2 m3 (x2[t] - x3[t]))/((x2[t] - x3[t])^2 + (y2[t] - y3[t])^2)^(3/2));
    eqn3ay = m3 y3''[t] ==  ((m1 m3 (y1[t] - y3[t]))/((x1[t] - x3[t])^2 + (y1[t] - y3[t])^2)^(3/2)) + ((m2 m3 (y2[t] - y3[t]))/((x2[t] - x3[t])^2 + (y2[t] - y3[t])^2)^(3/2));

    (*pos init*)
    eqn1xI = x1[0] == x1o[[1]];
    eqn1yI = y1[0] == x1o[[2]];

    eqn2xI = x2[0] == x2o[[1]];
    eqn2yI = y2[0] == x2o[[2]];

    eqn3xI = x3[0] == x3o[[1]];
    eqn3yI = y3[0] == x3o[[2]];

    (*vel init*)
    eqn1vxI = x1'[0] == v1o[[1]];
    eqn1vyI = y1'[0] == v1o[[2]];

    eqn2vxI = x2'[0] == v2o[[1]];
    eqn2vyI = y2'[0] == v2o[[2]];

    eqn3vxI = x3'[0] == v3o[[1]];
    eqn3vyI = y3'[0] == v3o[[2]];

    soln = NDSolve[{eqn1ax, eqn1ay, eqn2ax, eqn2ay, eqn3ax, eqn3ay, eqn1xI, eqn1yI, eqn2xI, eqn2yI, eqn3xI, eqn3yI, eqn1vxI, eqn1vyI, eqn2vxI, eqn2vyI, eqn3vxI, eqn3vyI}, {x1[t], y1[t], x2[t], y2[t], x3[t], y3[t]}, {t, 0, Tend}];

    mx1[t_] = x1[t] /. soln[[1]][[1]];
    my1[t_] = y1[t] /. soln[[1]][[2]];

    mx2[t_] = x2[t] /. soln[[1]][[3]];
    my2[t_] = y2[t] /. soln[[1]][[4]];

    mx3[t_] = x3[t] /. soln[[1]][[5]];
    my3[t_] = y3[t] /. soln[[1]][[6]];

    tStart = 0;
    tEnd = 0;

    tStart = RandomReal[{0, Tend}];
    tEnd = RandomReal[{0, Tend}];
    While[tStart > tEnd, tStart = RandomReal[{0, Tend}]; tEnd = RandomReal[{0, Tend}]];

    validationDatapoints = Append[validationDatapoints, {m1, m2, m3, tStart, tEnd, mx1[tStart], my1[tStart], mx2[tStart], my2[tStart], mx3[tStart], my3[tStart], mx1'[tStart], my1'[tStart], mx2'[tStart], my2'[tStart], mx3'[tStart], my3'[tStart], mx1[tEnd], my1[tEnd], mx2[tEnd], my2[tEnd], mx3[tEnd], my3[tEnd], mx1'[tEnd], my1'[tEnd], mx2'[tEnd], my2'[tEnd], mx3'[tEnd], my3'[tEnd]}];
    , 20]

Export["~/Desktop/nBodyNN/valData.csv", validationDatapoints, "CSV"]

(* traj = ParametricPlot[{{mx1[t], my1[t]}, {mx2[t], my2[t]}, {mx3[t], my3[t]}}, {t, 0, Tend}, PlotRange -> {{-10, 10}, {-10, 10}}] *)
(* Export["traj.pdf", traj] *)


