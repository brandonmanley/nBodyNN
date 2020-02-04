import os
import pandas as pd 
import math
import numpy as np 
import matplotlib.pyplot as plt
import random as r 
import preputil as util 

batch = 1
div_dir = "/Users/brandonmanley/Desktop/nBody/data/mathDiv/batchDiv"
div_df = util.concatCSV(div_dir, batch)

baryX = []
baryY = []

xList = []
yList = []

for trow in div_df.itertuples():
    baryX.append( ((trow.x1*trow.m1) + (trow.x2*trow.m2) + (trow.x3*trow.m3))/ (trow.m1 + trow.m2 + trow.m3) )
    baryY.append( ((trow.y1*trow.m1) + (trow.y2*trow.m2) + (trow.y3*trow.m3))/ (trow.m1 + trow.m2 + trow.m3) )

    xList.append(trow.x1)
    xList.append(trow.x2)
    xList.append(trow.x3)

    yList.append(trow.y1)
    yList.append(trow.y2)
    yList.append(trow.y3)


# plt.hexbin(baryX, baryY, gridsize=20)
plt.hexbin(xList, yList, gridsize=20, bins="log")
# plt.hist(xList, histtype="step", color="red")
# plt.hist(yList, histtype="step", color="blue")

# plt.xlabel("x [m]")
# plt.ylabel("y [m]")
plt.colorbar()

# draw boxes for coordinate origin spaces
sqX1 = [-3,-1,-1,-3,-3]
sqY1 = [1,1,2,2,1]

sqX3 = [3,5,5,3,3]
sqY3 = [3,3,5,5,3]

sqX2 = [0,2,2,0,0]
sqY2 = [-2,-2,0,0,-2]

plt.plot(sqX1, sqY1, color="red")
plt.plot(sqX2, sqY2, color="red")
plt.plot(sqX3, sqY3, color="red") 

plt.show()






