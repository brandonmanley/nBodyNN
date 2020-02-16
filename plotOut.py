import os
import pandas as pd 
import math
import numpy as np 
import matplotlib.pyplot as plt
import random as r 
import preputil as util 

def Extract(mylist, index): 
    return [item[index] for item in mylist] 

meta = "3_1"
in_df = util.prepData("/nBodyData/",meta)
print(in_df.head())
# pd.DataFrame.to_csv(in_df, "combined_"+meta+".csv")


r.seed(1)
indexArray = np.array(in_df['eventID'])
indexArray = np.unique((indexArray/ 10000).astype(int))
print(indexArray)
index = np.random.choice(indexArray)*10000
# index = 40000

eventNum = index
print('event:', int(eventNum/10000))

x1_i = []
x2_i = []
x3_i = []

x1_fSim = []
x2_fSim = []
x3_fSim = []

x1_fBrut = []
x2_fBrut = []
x3_fBrut = []

x1_fNN = []
x2_fNN = []
x3_fNN = []

tempdf = in_df.loc[(in_df['eventID'] >= eventNum) & (in_df['eventID'] <= (eventNum+2560))]
for temprow in tempdf.itertuples():
    if len(x1_i) < 1: 
        x1_i.append([temprow.x1, temprow.y1])
        x1_fSim.append([temprow.x1, temprow.y1])
        x1_fNN.append([temprow.x1, temprow.y1])
        x1_fBrut.append([temprow.x1, temprow.y1])
    if len(x2_i) < 1: 
        x2_i.append([temprow.x2, temprow.y2])
        x2_fSim.append([temprow.x2, temprow.y2])
        x2_fNN.append([temprow.x2, temprow.y2])
        x2_fBrut.append([temprow.x2, temprow.y2])
    if len(x3_i) < 1: 
        x3_i.append([temprow.x3, temprow.y3])
        x3_fSim.append([temprow.x3, temprow.y3])
        x3_fNN.append([temprow.x3, temprow.y3])
        x3_fBrut.append([temprow.x3, temprow.y3])
        

    x1_fSim.append([temprow.x1tEnd, temprow.y1tEnd])
    x2_fSim.append([temprow.x2tEnd, temprow.y2tEnd])
    x3_fSim.append([temprow.x3tEnd, temprow.y3tEnd])

    x1_fNN.append([temprow.x1tEnd_p, temprow.y1tEnd_p])
    x2_fNN.append([temprow.x2tEnd_p, temprow.y2tEnd_p])
    x3_fNN.append([temprow.x3tEnd_p, temprow.y3tEnd_p]) 

    x1_fBrut.append([temprow.x1tEnd_b, temprow.y1tEnd_b])
    x2_fBrut.append([temprow.x2tEnd_b, temprow.y2tEnd_b])
    x3_fBrut.append([temprow.x3tEnd_b, temprow.y3tEnd_b]) 

p1x_sim = Extract(x1_fSim, 0)
p1y_sim = Extract(x1_fSim, 1)
p2x_sim = Extract(x2_fSim, 0)
p2y_sim = Extract(x2_fSim, 1)
p3x_sim = Extract(x3_fSim, 0)
p3y_sim = Extract(x3_fSim, 1)

p1x_NN = Extract(x1_fNN, 0)
p1y_NN = Extract(x1_fNN, 1)
p2x_NN = Extract(x2_fNN, 0)
p2y_NN = Extract(x2_fNN, 1)
p3x_NN = Extract(x3_fNN, 0)
p3y_NN = Extract(x3_fNN, 1)

p1x_Brut = Extract(x1_fBrut, 0)
p1y_Brut = Extract(x1_fBrut, 1)
p2x_Brut = Extract(x2_fBrut, 0)
p2y_Brut = Extract(x2_fBrut, 1)
p3x_Brut = Extract(x3_fBrut, 0)
p3y_Brut = Extract(x3_fBrut, 1)


df1_sim = pd.DataFrame({'Sim x1':p1x_sim, 'Sim y1': p1y_sim})
df2_sim = pd.DataFrame({'Sim x2':p2x_sim, 'Sim y2': p2y_sim})
df3_sim = pd.DataFrame({'Sim x3':p3x_sim, 'Sim y3': p3y_sim})

df1_Brut = pd.DataFrame({'Brut x1':p1x_Brut, 'Brut y1': p1y_Brut})
df2_Brut = pd.DataFrame({'Brut x2':p2x_Brut, 'Brut y2': p2y_Brut})
df3_Brut = pd.DataFrame({'Brut x3':p3x_Brut, 'Brut y3': p3y_Brut})

# print(df1_Brut.head())

df1_NN = pd.DataFrame({'NN x1':p1x_NN, 'NN y1': p1y_NN})
df2_NN = pd.DataFrame({'NN x2':p2x_NN, 'NN y2': p2y_NN})
df3_NN = pd.DataFrame({'NN x3':p3x_NN, 'NN y3': p3y_NN})

p1_i = pd.DataFrame({'Initial x1': [x1_i[0][0]], 'Initial y1': [x1_i[0][1]]})
p2_i = pd.DataFrame({'Initial x2': [x2_i[0][0]], 'Initial y2': [x2_i[0][1]]})
p3_i = pd.DataFrame({'Initial x3': [x3_i[0][0]], 'Initial y3': [x3_i[0][1]]})

# print(p1_i, p2_i, p3_i)
tsize = 0
plt.plot('Sim x1', 'Sim y1', data=df1_sim, color='blue', marker='o',linewidth=1, markersize=tsize, markevery=30)
plt.plot('Sim x2', 'Sim y2', data=df2_sim, color='red', marker='o',linewidth=1, markersize=tsize, markevery=30)
plt.plot('Sim x3', 'Sim y3', data=df3_sim, color='green', marker='o',linewidth=1, markersize=tsize, markevery=30)

plt.plot('Brut x1', 'Brut y1', data=df1_Brut, color='blue', marker='x', linestyle='dashed', linewidth=1, markersize=tsize, markevery=30)
plt.plot('Brut x2', 'Brut y2', data=df2_Brut, color='red', marker='x', linestyle='dashed', linewidth=1, markersize=tsize, markevery=30)
plt.plot('Brut x3', 'Brut y3', data=df3_Brut, color='green', marker='x', linestyle='dashed', linewidth=1, markersize=tsize, markevery=30)

# plt.plot('NN x1', 'NN y1', data=df1_NN, color='blue', marker='x', linestyle=':', linewidth=1, markersize=tsize, markevery=30)
# plt.plot('NN x2', 'NN y2', data=df2_NN, color='red', marker='x', linestyle=':', linewidth=1, markersize=tsize, markevery=30)
# plt.plot('NN x3', 'NN y3', data=df3_NN, color='green', marker='x', linestyle=':', linewidth=1, markersize=tsize, markevery=30)

plt.plot('Initial x1', 'Initial y1', data=p1_i, color='black', marker='s')
plt.plot('Initial x2', 'Initial y2', data=p2_i, color='black', marker='s')
plt.plot('Initial x3', 'Initial y3', data=p3_i, color='black', marker='s')

plt.legend(loc='best', ncol=3, fancybox=True)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Event {0}'.format(int(eventNum/10000)))
plt.show()
# plt.savefig('event_{0}_paths.png'.format(eventNum))


# plot error as a function of bary
# bary = []
# timelist = []
# elist = []

# for row in in_df.itertuples():
#     bary.append(((row.m1*row.x1)+(row.m2*row.x2)+(row.m3*row.x3))/(row.m1+row.m2+row.m3))

#     timelist.append(row.tEnd)

#     e1 = math.sqrt(((row.x1tEnd - row.x1tEnd_2)**2)+((row.y1tEnd - row.y1tEnd_2)**2))
#     e2 = math.sqrt(((row.x2tEnd - row.x2tEnd_2)**2)+((row.y2tEnd - row.y2tEnd_2)**2))
#     e3 = math.sqrt(((row.x3tEnd - row.x3tEnd_2)**2)+((row.y3tEnd - row.y3tEnd_2)**2)) 
#     elist.append(math.sqrt((e1**2)+(e2**2)+(e3**2)))


# plt.scatter(timelist, elist, c='blue', s=1)
# # plt.scatter(r2, e2, c='red', s=1)
# # plt.scatter(r3, e3, c='green', s=1)
# plt.show()



