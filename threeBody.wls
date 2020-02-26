#!/usr/bin/env wolframscript

batchNum = 5;
iFile = 0;
If[Length[$ScriptCommandLine]<2, Print["Usage: fileNum"]; Quit[], iFile=$ScriptCommandLine[[2]]]
Print["Starting file: ", iFile];

inputFile = "/nBodyData/inputs/indat_"<>ToString[batchNum]<>"_" <> ToString[iFile] <> ".dat";
basePath = "/nBodyData/mathSim/";


(* nEventsPerFile = 1;  *)
timeStampsPerEvent = 2560;
Tend = 10.0;
Tstart = 0;
divergentDatapoints = 0;
divergentEventCutOff = 10^5;
delta = 0.01;
dateCreated = DateString["ISODate"];

(* divData = {};
divData = Append[divData, {"m1", "m2", "m3", "x1", "x2", "x3", "y1", "y2", "y3", "tEnd"}]; *)

eventID = 10000;
events = 0;

Print["Input file: ", inputFile];
input = Import[inputFile,{"CSV", "RawData"}];

fileString = "batch" <> ToString[batchNum] <> "_" <> ToString[iFile] <> ".csv";
exportString = basePath <> fileString;
DeleteFile[exportString];
Print["Exported path: ", exportString];
(* Export[exportString, validationDatapoints, "CSV"]; *)
validationDatapoints = {};
validationDatapoints = Append[validationDatapoints, {"eventID", "m1", "m2", "m3", "x1", "x2", "x3", "y1", "y2", "y3", "tEnd", "x1tEnd", "x2tEnd", "x3tEnd", "y1tEnd", "y2tEnd", "y3tEnd" }];
headerStr = OpenAppend[exportString];
WriteString[headerStr, ExportString[validationDatapoints, "CSV"]];
Close[headerStr];

nDataPoints = Length[input]*timeStampsPerEvent;
If[nDataPoints < 1, Print["Error: no data input"]; Quit[]]
Print["Making ", nDataPoints, " total data points"];
WriteString[$Output, "# of events (1000s): "];

For[iEvent=0, iEvent < Length[input], iEvent += 1;

    flag = False;
    If[iEvent > 9, flag = True];
    If[Mod[iEvent, 10] == 0, WriteString[$Output, ToString[iEvent/10]]]; 

    (*
    v indices: 
    0: m1    1: p1x  2: p1y    3: p1z   4: p1vx   5: p1vy   6: p1vz
    7: m2    8: p2x  9: p2y   10: p2z  11: p2vx  12: p2vy  13: p2vz
    14: m3  15: p3x  16: p3y  17: p3z  18: p3vx  19: p3vy  20: p3vz

    data indices: shifted up 1
    *)
    
    (* x1o = {RandomReal[{-3, -1}], RandomReal[{1, 2}]};
    x2o = {RandomReal[{0, 2}], RandomReal[{-2, 0}]};
    x3o = {RandomReal[{3, 5}], RandomReal[{3, 5}]}; *)
    x1o = {ToExpression[input[[iEvent]][[2]]], ToExpression[input[[iEvent]][[3]]]};
    x2o = {ToExpression[input[[iEvent]][[9]]], ToExpression[input[[iEvent]][[10]]]};
    x3o = {ToExpression[input[[iEvent]][[16]]], ToExpression[input[[iEvent]][[17]]]};

    v1o = {ToExpression[input[[iEvent]][[5]]], ToExpression[input[[iEvent]][[6]]]};
    v2o = {ToExpression[input[[iEvent]][[12]]], ToExpression[input[[iEvent]][[13]]]};
    v3o = {ToExpression[input[[iEvent]][[19]]], ToExpression[input[[iEvent]][[20]]]};

    (* m1 = 5;
    m2 = 12;
    m3 = 13; *)
    m1 = ToExpression[input[[iEvent]][[1]]];
    m2 = ToExpression[input[[iEvent]][[8]]];
    m3 = ToExpression[input[[iEvent]][[15]]];

    G =  6.674*(10**-11);

    (*f=ma*)
    eqn1ax = m1 x1''[t] ==  ((m1 m2 (x2[t] - x1[t]))/((x1[t] - x2[t])^2 + (y2[t] - y1[t])^2)^(3/2)) + ((m1 m3 (x3[t] - x1[t]))/((x1[t] - x3[t])^2 + (y1[t] - y3[t])^2)^(3/2));
    eqn1ay = m1 y1''[t] ==  ((m1 m2 (y2[t] - y1[t]))/((x1[t] - x2[t])^2 + (y2[t] - y1[t])^2)^(3/2)) + ((m1 m3 (y3[t] - y1[t]))/((x1[t] - x3[t])^2 + (y1[t] - y3[t])^2)^(3/2));
    
    eqn2ax = m2 x2''[t] ==  ((m1 m2 (x1[t] - x2[t]))/((x1[t] - x2[t])^2 + (y1[t] - y2[t])^2)^(3/2)) + ((m2 m3 (x3[t] - x2[t]))/((x2[t] - x3[t])^2 + (y2[t] - y3[t])^2)^(3/2));
    eqn2ay = m2 y2''[t] ==  ((m1 m2 (y1[t] - y2[t]))/((x1[t] - x2[t])^2 + (y1[t] - y2[t])^2)^(3/2)) + ((m2 m3 (y3[t] - y2[t]))/((x2[t] - x3[t])^2 + (y2[t] - y3[t])^2)^(3/2));

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

    soln = Quiet[NDSolve[{ eqn1ax, eqn1ay, eqn2ax, 
    eqn2ay, eqn3ax, eqn3ay, eqn1xI, eqn1yI, eqn2xI, eqn2yI, eqn3xI, 
    eqn3yI, eqn1vxI, eqn1vyI, eqn2vxI, eqn2vyI, eqn3vxI, 
    eqn3vyI}, {x1[t], y1[t], x2[t], y2[t], x3[t], y3[t]}, {t, 0, 
    Tend}]];

    mx1[t_] = x1[t] /. soln[[1]][[1]];
    my1[t_] = y1[t] /. soln[[1]][[2]];

    mx2[t_] = x2[t] /. soln[[1]][[3]];
    my2[t_] = y2[t] /. soln[[1]][[4]];

    mx3[t_] = x3[t] /. soln[[1]][[5]];
    my3[t_] = y3[t] /. soln[[1]][[6]];

    tEndtime = Subdivide[Tstart, Tend, timeStampsPerEvent];

    thisEvent = {};

    (* If[flag, Print["hmm2"]]; *)
    For[i=1, i < Length[tEndtime]+1,

        (* If[isDivEvent, Break]; *)

        Quiet[
            point = {eventID, m1, m2, m3, x1o[[1]], x2o[[1]], x3o[[1]], x1o[[2]], x2o[[2]], x3o[[2]], 
                tEndtime[[i]], mx1[tEndtime[[i]]], mx2[tEndtime[[i]]], mx3[tEndtime[[i]]], 
                my1[tEndtime[[i]]], my2[tEndtime[[i]]], my3[tEndtime[[i]]] };

            testPoint = {mx1[tEndtime[[i]]], mx2[tEndtime[[i]]], mx3[tEndtime[[i]]], 
                my1[tEndtime[[i]]], my2[tEndtime[[i]]], my3[tEndtime[[i]]] };
        
            (* divDatapoint = {m1, m2, m3, x1o[[1]], x2o[[1]], x3o[[1]], x1o[[2]], x2o[[2]], x3o[[2]]}; *)

            (* ex: 1109000002560 -> step 2560 of event 900000 of file 1 of batch 1 *)
            (* globalID = 1e9*batchNum + 1e8*fileNum + 1e4*iE *)
            point[[1]] = (10^9)*batchNum + (10^8)*ToExpression[iFile] + eventID+i; 
            (* If[flag, Print[point]]; *)

            If[
                Length[Select[# > divergentEventCutOff&][testPoint]]>0, ++i; Break, thisEvent = Append[thisEvent,point]
            ]; 
            (* If[flag, Print[point]]; *)

            ]; ++i;
        ];

    str = OpenAppend[exportString];
    WriteString[str, ExportString[thisEvent, "CSV"]];
    Close[str];

    events += 1;
    eventID += 10000;

];
Print["\n"];



(* Print[Length[divData]];
baseDivPath = "~/Desktop/nBody/data/mathDiv/batchDiv";
exportDivString = baseDivPath <> ToString[batchNum] <> "_" <> ToString[iFile] <> ".csv";
Export[exportDivString, divData, "CSV"];
Run["cp "<>exportDivString<> " ~/Desktop/OneDrive\ -\ The\ Ohio\ State\ University/nBody\ Data/mathDiv/"]; *)


(* Make meta data
baseMetaPath = "/nBodydata/mathMeta/";
metaFileString = "batch" <> ToString[batchNum] <> "_" <> ToString[iFile] <> "_meta.json";
metaDataString = baseMetaPath <> metaFileString;
Export[metaDataString, {"Batch number"->batchNum, "File number"->iFile, "Events"->nEventsPerFile, "Time stamps per event"->timeStampsPerEvent, 
                        "Data points"->Length[validationDatapoints], "Time start"->Tstart, "Time end"->Tend, 
                        "Divergent data points"->divergentDatapoints, "Divergent event cut off"->divergentEventCutOff, 
                        "Delta"->delta, "Date created"->dateCreated}]; *)
