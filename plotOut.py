import os
import pandas as pd 
import math

meta = "504_2020-01-27"
in_file = "val_"+meta+".csv"
pred_file = "predicted_paths_"+meta+".csv"

pred_df = pd.read_csv(pred_file, header=None)
pred_df.rename(columns={0: 'x1tEnd', 1: 'x2tEnd', 2: 'x3tEnd', 3: 'y1tEnd', 4: 'y2tEnd', 5: 'y3tEnd', 6: 'eventID'}, inplace=True)
# pred_df.to_csv(pred_file, index=False)
in_df = pd.read_csv(in_file)
pred_df.eventID = pred_df.eventID.astype(int)
pred_df.sort_values(["eventID"], axis=0, ascending=True, inplace= True) 

for col in ['x1tEnd', 'x2tEnd', 'x3tEnd', 'y1tEnd', 'y2tEnd', 'y3tEnd']:
    in_df[col+'_2'] = in_df['eventID'].map(pred_df.set_index('eventID')[col])

x1_i = []
x2_i = []
x3_i = []
y1_i = []
y2_i = []
y3_i = []

x1_fSim = []
x1_fNN = []
x2_fSim = []
x2_fNN = []
x3_fSim = []
x3_fNN = []
y1_fSim = []
y1_fNN = []
y2_fSim = []
y2_fNN = []
y3_fSim = []
y3_fNN = []


previousEventNum = 0
nEvents = 0

for row in in_df.itertuples():
    if math.isnan(row.x1tEnd_2): continue 
    eventNum = row.eventID
    if eventNum != 2905: continue

    eventNum = int(eventNum/100)*100
    print(eventNum)
    
    print(in_df.loc[(in_df['eventID'] >= eventNum) & (in_df['eventID'] <= (eventNum+100))])

        


    # print(type(row.x1tEnd_2))



# print(pred_df.head())
# print(in_df.head())

# for index, row in in_df.head().iterrows():
#     e_id = int(row["eventID"])
#     print(pred_df.loc[pred_df["eventID"] == e_id])
