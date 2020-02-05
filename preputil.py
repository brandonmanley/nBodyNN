import os
import pandas as pd 
import glob

def prepData(dir, meta):
    math_file = dir+"mathSim/batch"+meta+".csv"
    brut_file = dir+"brutusSim/batch_brutus"+meta+".csv"

    # pred_file = "predicted_paths"+meta+".csv"

    # read sim/prediction csvs and label the columns of prediction
    # pred_df = pd.read_csv(pred_file, header=None)
    # pred_df.rename(columns={0: 'x1tEnd', 1: 'x2tEnd', 2: 'x3tEnd', 3: 'y1tEnd', 4: 'y2tEnd', 5: 'y3tEnd', 6: 'eventID'}, inplace=True)
    
    math_df = pd.read_csv(math_file)
    brut_df = pd.read_csv(brut_file)

    # print(math_df.head())
    # print("\n")
    # print(brut_df.head())

    # for row in brut_df.itertuples():
    #     print(row)
    # set event id to be an int and sort by event id
    # pred_df.eventID = pred_df.eventID.astype(int)
    # pred_df.sort_values(["eventID"], axis=0, ascending=True, inplace= True) 
    brut_df["eventID"] = pd.to_numeric(brut_df["eventID"])
    
    for i in range(1,4):
        brut_df["x{0}".format(i)] = pd.to_numeric(brut_df["x{0}".format(i)])
        brut_df["y{0}".format(i)] = pd.to_numeric(brut_df["y{0}".format(i)])
        brut_df["x{0}tEnd".format(i)] = pd.to_numeric(brut_df["x{0}tEnd".format(i)])
        brut_df["y{0}tEnd".format(i)] = pd.to_numeric(brut_df["y{0}tEnd".format(i)])

    brut_df.sort_values(["eventID"], axis=0, ascending=True, inplace= True) 

    # combine data frames into 1
    for col in ['x1tEnd', 'x2tEnd', 'x3tEnd', 'y1tEnd', 'y2tEnd', 'y3tEnd']:
        math_df[col+'_b'] = math_df['eventID'].map(brut_df.set_index('eventID')[col])
        # math_df[col+'_p'] = math_df['eventID'].map(pred_df.set_index('eventID')[col])

    # remove any null predictions
    return math_df[pd.notnull(math_df["x1tEnd_b"])]
    # math_df.to_csv("data/prepared_"+meta+".csv")


def concatCSV(dir, batch):
    extension = 'csv'
    all_filenames = [i for i in glob.glob(dir+'{0}_*.{1}'.format(batch,extension))]
    return pd.concat([pd.read_csv(f) for f in all_filenames ])
    # combined_csv.to_csv(workDir+"combined_data.csv", index=False, encoding='utf-8-sig')

