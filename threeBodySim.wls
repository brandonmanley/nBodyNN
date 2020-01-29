#!/usr/bin/env wolframscript

nIntervals = RandomInteger[{5, 12}];
Print["Using ", nIntervals, " time stamps"];

delta = 0.01;

validationDatapoints = {};
validationDatapoints = Append[validationDatapoints, {"eventID", "m1", "m2", "m3", "x1", "x2", "x3", "y1", "y2", "y3", "tEnd", "x1tEnd", "x2tEnd", "x3tEnd", "y1tEnd", "y2tEnd", "y3tEnd" }];
   
nDataPoints = 10000;
eventID = 100;

count = 0;
Print["Making ", nDataPoints, " total data points"];
WriteString[$Output, "# of events (100s): "];

    While[ Length[validationDatapoints] < nDataPoints +1,

    count += 1;
    If[Mod[count, 100] == 0, WriteString[$Output, ToString[count/100]]];
  
    x1o = {RandomReal[{-3, -1}], RandomReal[{1, 2}]};
    x2o = {RandomReal[{0, 2}], RandomReal[{-2, 0}]};
    x3o = {RandomReal[{3, 5}], RandomReal[{3, 5}]};

    v1o = {0,0};
    v2o = {0,0};
    v3o = {0,0};

    Tend = 110.0;

    m1 = 5;
    m2 = 12;
    m3 = 13;

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

    soln = NDSolve[{ 
   WhenEvent[Abs[((x1[t] - x3[t])^2 + (y1[t] - y3[t])^2)] < delta, 
    "StopIntegration"], 
   WhenEvent[Abs[((x2[t] - x3[t])^2 + (y2[t] - y3[t])^2)] < delta, 
    "StopIntegration"], 
   WhenEvent[Abs[((x1[t] - x2[t])^2 + (y1[t] - y2[t])^2)] < delta, 
    "StopIntegration"], eqn1ax, eqn1ay, eqn2ax, 
   eqn2ay, eqn3ax, eqn3ay, eqn1xI, eqn1yI, eqn2xI, eqn2yI, eqn3xI, 
   eqn3yI, eqn1vxI, eqn1vyI, eqn2vxI, eqn2vyI, eqn3vxI, 
   eqn3vyI}, {x1[t], y1[t], x2[t], y2[t], x3[t], y3[t]}, {t, 0, 
   Tend}];

    mx1[t_] = x1[t] /. soln[[1]][[1]];
    my1[t_] = y1[t] /. soln[[1]][[2]];

    mx2[t_] = x2[t] /. soln[[1]][[3]];
    my2[t_] = y2[t] /. soln[[1]][[4]];

    mx3[t_] = x3[t] /. soln[[1]][[5]];
    my3[t_] = y3[t] /. soln[[1]][[6]];

    (* tEndtime = Subdivide[1, Tend, nIntervals]; *)
    tEndtime = {0.1, 10, 20, 30};

    For[i=1, i < Length[tEndtime]+1,
        Quiet[
            point = {eventID, m1, m2, m3, x1o[[1]], x1o[[2]], x2o[[1]], x2o[[2]], x3o[[1]], x3o[[2]], tEndtime[[i]], mx1[tEndtime[[i]]], mx2[tEndtime[[i]]], mx3[tEndtime[[i]]], my1[tEndtime[[i]]], my2[tEndtime[[i]]], my3[tEndtime[[i]]] };
            If[Length[Select[# > 10^6&][point] ]>0, ++i; Continue, point[[1]] = eventID+i; validationDatapoints = Append[validationDatapoints,point];]
            ]; ++i;
        ];
    eventID += 100;
    ]

Print["\n"];

basePath = "~/Desktop/nBodyNN/data/varyNintervals/val_"
exportString = basePath <> ToString[nIntervals] <> "_" <> ToString[Length[validationDatapoints]] <> "_" <> DateString["ISODate"] <> ".csv"
Print["Exported path: ", exportString]
Export[exportString, validationDatapoints, "CSV"]