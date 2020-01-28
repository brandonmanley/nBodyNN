import os
import pandas as pd 
import math
import numpy as np 
import matplotlib.pyplot as plt
import random as r 

def Extract(mylist, index): 
    return [item[index] for item in mylist] 

meta = "504_2020-01-27"
nEvents = int(int(meta[0:meta.find('_')])/6)
in_file = "val_"+meta+".csv"
pred_file = "predicted_paths_"+meta+".csv"

# read sim/prediction csvs and label the columns of prediction
pred_df = pd.read_csv(pred_file, header=None)
pred_df.rename(columns={0: 'x1tEnd', 1: 'x2tEnd', 2: 'x3tEnd', 3: 'y1tEnd', 4: 'y2tEnd', 5: 'y3tEnd', 6: 'eventID'}, inplace=True)
in_df = pd.read_csv(in_file)

# set event id to be an int and sort by event id
pred_df.eventID = pred_df.eventID.astype(int)
pred_df.sort_values(["eventID"], axis=0, ascending=True, inplace= True) 

# combine data frames into 1
for col in ['x1tEnd', 'x2tEnd', 'x3tEnd', 'y1tEnd', 'y2tEnd', 'y3tEnd']:
    in_df[col+'_2'] = in_df['eventID'].map(pred_df.set_index('eventID')[col])

# remove any null predictions
in_df = in_df[pd.notnull(in_df["x1tEnd_2"])]

r.seed(r.randint(1,len(in_df.index)))
index = r.randint(1,len(in_df.index))
eventNum = in_df.iloc[index].eventID
print(eventNum)


x1_i = []
x2_i = []
x3_i = []

x1_fSim = []
x1_fNN = []
x2_fSim = []
x2_fNN = []
x3_fSim = []
x3_fNN = []

tempdf = in_df.loc[(in_df['eventID'] >= eventNum) & (in_df['eventID'] <= (eventNum+100))]
for temprow in tempdf.itertuples():
    if len(x1_i) < 1: 
        x1_i.append([temprow.x1, temprow.y1])
        x1_fSim.append([temprow.x1, temprow.y1])
        x1_fNN.append([temprow.x1, temprow.y1])
    if len(x2_i) < 1: 
        x2_i.append([temprow.x2, temprow.y2])
        x2_fSim.append([temprow.x2, temprow.y2])
        x2_fNN.append([temprow.x2, temprow.y2])
    if len(x3_i) < 1: 
        x3_i.append([temprow.x3, temprow.y3])
        x3_fSim.append([temprow.x3, temprow.y3])
        x3_fNN.append([temprow.x3, temprow.y3])
        

    x1_fSim.append([temprow.x1tEnd, temprow.y1tEnd])
    x2_fSim.append([temprow.x2tEnd, temprow.y2tEnd])
    x3_fSim.append([temprow.x3tEnd, temprow.y3tEnd])

    x1_fNN.append([temprow.x1tEnd_2, temprow.y1tEnd_2])
    x2_fNN.append([temprow.x2tEnd_2, temprow.y2tEnd_2])
    x3_fNN.append([temprow.x3tEnd_2, temprow.y3tEnd_2]) 

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

df1_sim = pd.DataFrame({'Sim x1':p1x_sim, 'Sim y1': p1y_sim})
df2_sim = pd.DataFrame({'Sim x2':p2x_sim, 'Sim y2': p2y_sim})
df3_sim = pd.DataFrame({'Sim x3':p3x_sim, 'Sim y3': p3y_sim})

df1_NN = pd.DataFrame({'NN x1':p1x_NN, 'NN y1': p1y_NN})
df2_NN = pd.DataFrame({'NN x2':p2x_NN, 'NN y2': p2y_NN})
df3_NN = pd.DataFrame({'NN x3':p3x_NN, 'NN y3': p3y_NN})

p1_i = pd.DataFrame({'Initial x1': [x1_i[0][0]], 'Initial y1': [x1_i[0][1]]})
p2_i = pd.DataFrame({'Initial x2': [x2_i[0][0]], 'Initial y2': [x2_i[0][1]]})
p3_i = pd.DataFrame({'Initial x3': [x3_i[0][0]], 'Initial y3': [x3_i[0][1]]})

plt.plot('Sim x1', 'Sim y1', data=df1_sim, color='blue', marker='o')
plt.plot('Sim x2', 'Sim y2', data=df2_sim, color='red', marker='o')
plt.plot('Sim x3', 'Sim y3', data=df3_sim, color='green', marker='o')

plt.plot('NN x1', 'NN y1', data=df1_NN, color='blue', linestyle='dashed', marker='v')
plt.plot('NN x2', 'NN y2', data=df2_NN, color='red', linestyle='dashed', marker='v')
plt.plot('NN x3', 'NN y3', data=df3_NN, color='green',linestyle='dashed', marker='v')

plt.plot('Initial x1', 'Initial y1', data=p1_i, color='black', marker='s')
plt.plot('Initial x2', 'Initial y2', data=p2_i, color='black', marker='s')
plt.plot('Initial x3', 'Initial y3', data=p3_i, color='black', marker='s')

plt.legend(loc='best', ncol=3, fancybox=True)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Event {0}'.format(int(eventNum/100)))
plt.show()
# plt.savefig('event_{0}_paths.png'.format(eventNum))

