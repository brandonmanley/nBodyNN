import pandas as pd 
import preputil as prep 

df = prep.concatCSV("/nBodyData/mathSim/batch3")
print(len(df.index))