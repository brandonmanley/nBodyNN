import os
import glob
import pandas as pd

workDir = "/Users/brandonmanley/Desktop/nBody/data/mathSim/"
extension = 'csv'
all_filenames = [i for i in glob.glob(workDir+'*.{}'.format(extension))]
print(all_filenames)

combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv(workDir+"combined_data.csv", index=False, encoding='utf-8-sig')