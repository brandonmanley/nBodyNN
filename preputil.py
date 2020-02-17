import os
import pandas as pd 
import glob

def prepData(dir, meta):  # prep data for analysis (enforce headers, ensure column types are correct, ensure sim/pred match up)
    prediction = False

    math_file = dir+"mathSim/batch"+meta+".csv"
    # math_file = dir+"julusSim/batch_brutus"+meta+".csv"
    jul_file = dir+"julSim/julia_batch"+meta+".csv"
    pred_file = dir+"pred/predicted_paths_"+meta+".csv"

    # read sim/prediction csvs and label the columns of prediction
    pred_df = pd.read_csv(pred_file, header=None)
    math_df = pd.read_csv(math_file)
    jul_df = pd.read_csv(jul_file, header=None)

    pred_df.rename(columns={0: 'x1tEnd', 1: 'x2tEnd', 2: 'x3tEnd', 3: 'y1tEnd', 4: 'y2tEnd', 5: 'y3tEnd', 6: 'eventID'}, inplace=True)
    jul_df.rename(columns={0: 'eventID', 1:'m1', 2:'m2', 3:'m3', 4: 'x1', 5: 'x2', 6: 'x3', 7: 'y1', 8: 'y2', 9: 'y3', 10:'tEnd', 11: 'x1tEnd', 12: 'x2tEnd', 13: 'x3tEnd', 14: 'y1tEnd', 15: 'y2tEnd', 16: 'y3tEnd'}, inplace=True)
    
   
    # print(math_df.head())
    # print("\n")
    # print(jul_df.head())

    # for row in jul_df.itertuples():
    #     print(row)


    # set event id to be an int and sort by event id
    # pred_df.eventID = pred_df.eventID.astype(int)
    # pred_df.sort_values(["eventID"], axis=0, ascending=True, inplace= True) 
    jul_df["eventID"] = pd.to_numeric(jul_df["eventID"])
    
    # manually set columns to be float type 
    for i in range(1,4):
        jul_df["x{0}".format(i)] = pd.to_numeric(jul_df["x{0}".format(i)])
        jul_df["y{0}".format(i)] = pd.to_numeric(jul_df["y{0}".format(i)])
        jul_df["x{0}tEnd".format(i)] = pd.to_numeric(jul_df["x{0}tEnd".format(i)])
        jul_df["y{0}tEnd".format(i)] = pd.to_numeric(jul_df["y{0}tEnd".format(i)])

    # sort by eventID
    jul_df.sort_values(["eventID"], axis=0, ascending=True, inplace= True) 

    # combine data frames into 1
    for col in ['x1tEnd', 'x2tEnd', 'x3tEnd', 'y1tEnd', 'y2tEnd', 'y3tEnd']:
        math_df[col+'_b'] = math_df['eventID'].map(jul_df.set_index('eventID')[col])
        # math_df[col+'_p'] = math_df['eventID'].map(pred_df.set_index('eventID')[col])

    # remove any null predictions / matching with other sim
    math_df = math_df[pd.notnull(math_df["x1tEnd_b"])]
    # math_df = math_df[pd.notnull(math_df["x1tEnd_p"])]
    
    return math_df
    # math_df.to_csv("data/prepared_"+meta+".csv")





def concatCSV(filename): # returns a fully joined pandas df for a single batch
    all_filenames = [i for i in glob.glob(dir+'*.csv']
    return pd.concat([pd.read_csv(f) for f in all_filenames ])
    # combined_csv.to_csv(workDir+"combined_data.csv", index=False, encoding='utf-8-sig')

