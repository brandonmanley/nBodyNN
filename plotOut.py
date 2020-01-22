import os
import pandas as pd 

fname = "~/Desktop/nBodyNN/"
df = pd.read_csv(fname)
df.columns = ['id','m1', 'm2', 'm3', 
't_f',
'x1_i','x2_i','x3_i','y1_i','y2_i','y3_i',
'x1_f','x2_f','x3_f','y1_f','y2_f','y3_f'
]

df.head()