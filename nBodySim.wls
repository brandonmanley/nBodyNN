#!/usr/bin/env wolframscript

nMasses = 10 (* grab from command line*)
tEnd = 0.1

massList = {}
x0List = {}
y0List = {}

vx0List = {}
vy0List = {}

(*init params*)
For[i = 0, i < nMasses, i++, massList = Append[massList, RandomReal[{1, 10}]]]
For[i = 0, i < nMasses, i++, x0List = Append[x0List, RandomReal[{-1, 1}]]]
For[i = 0, i < nMasses, i++, y0List = Append[y0List, RandomReal[{-1, 1}]]]
For[i = 0, i < nMasses, i++, vx0List = Append[vx0List, RandomReal[{-1, 1}]]]
For[i = 0, i < nMasses, i++, vy0List = Append[vy0List, RandomReal[{-1, 1}]]]

xList[n_] := Array[Symbol["x" <> ToString[#]][t] &, n]
yList[n_] := Array[Symbol["y" <> ToString[#]][t] &, n]









