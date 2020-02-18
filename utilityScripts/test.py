import pandas as pd 
import preputil as prep 

meta = "3_1"
df = prep.prepData("/nBodyData/", meta)
# df = prep.concatCSV("/nBodyData/mathSim/batch3")
# print(len(df.index))