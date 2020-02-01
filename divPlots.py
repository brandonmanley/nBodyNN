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

for trow in div_df.itertuples():
    baryX.append( ((trow.x1*trow.m1) + (trow.x2*trow.m2) + (trow.x3*trow.m3))/ (trow.m1 + trow.m2 + trow.m3) )
    baryY.append( ((trow.y1*trow.m1) + (trow.y2*trow.m2) + (trow.y3*trow.m3))/ (trow.m1 + trow.m2 + trow.m3) )

bary_div = pd.DataFrame({'xDiv':baryX, 'yDiv': baryY})
fig, axs = plt.subplots(figsize=(7, 5), sharex=True, sharey=True,tight_layout=True)
h =axs.hexbin("xDiv", "yDiv", data=bary_div, gridsize=20, bins="log")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
fig.colorbar(h, ax=axs)

plt.show()






