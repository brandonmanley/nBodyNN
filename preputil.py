import os
import pandas as pd 

def prepData(meta):
    in_file = "data/val_"+meta+".csv"
    pred_file = "data/predicted_paths_"+meta+".csv"

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
    return in_df[pd.notnull(in_df["x1tEnd_2"])]
    # in_df.to_csv("data/prepared_"+meta+".csv")

