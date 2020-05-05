#Using imput data of three body motion, trains neural network to predict
#the positions of the bodies at a given time using initial positions
#By Luca Lavezzo, Brandon Manley, Jan. 2020

import pandas as pd
from plotly.offline import iplot
import plotly.graph_objs as go
import numpy as np
from sklearn import metrics
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from itertools import repeat
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation

workDir = "/users/PAS1585/llavez99/"

#Import data
dataPath = workDir + "data/nbody/5body/"
        
# df1 = pd.read_csv(dataPath+"brutus10_1_4.csv")
# df2 = pd.read_csv(dataPath+"brutus10_2_4.csv")
# df = pd.concat([df1,df2])

# df_test = df1.sample(2560*10)

# i_col = []
# o_col = []
# colNames = ["m","x", "y", "dx", "dy"]
# nBodies = 4
# for col in colNames:
#     for n in range(1, nBodies+1):
#         i_col.append(col+str(n))
#         if col != "m":
#             o_col.append(col+"f"+str(n))
# i_col.append("t")

# df_y_test = df_test[o_col]
# df_X_test = df_test[i_col]
# X_test = df_X_test.to_numpy()
# y_test = df_y_test.to_numpy()

df_X_test = pd.read_csv(dataPath+"X_test_5body.csv")

X_test = df_X_test.to_numpy()

df1 = pd.read_csv(dataPath+"brutus10_1_5.csv")
df2 = pd.read_csv(dataPath+"brutus10_2_5.csv")
df = pd.concat([df1,df2])

test_rows = []
eps = 0.00001
X_test = X_test.astype('float64')
df = df.astype('float64')
print(df.head())
print(X_test)
print(X_test.shape)
for row in X_test:

    i = df.loc[(df['m1'] < row[0] + eps) & (df['m1'] > row[0] - eps) & (df['t'] < row[25] + eps) & (df['t'] > row[25] - eps)].index[0]
    test_rows.append(i)

dfTest = df.iloc[test_rows,:] 

i_col = []
o_col = []
colNames = ["m","x", "y", "dx", "dy"]
nBodies = 5
for col in colNames:
    for n in range(1, nBodies+1):
        i_col.append(col+str(n))
        if col != "m":
            o_col.append(col+"f"+str(n))
i_col.append("t")
    
df_y_test = dfTest[o_col]
df_X_test = dfTest[i_col]
X_test = df_X_test.to_numpy()
y_test = df_y_test.to_numpy()

#Run the neural network with the best number of hidden nodes and epochs
hidden_nodes = 300   
n_epochs = 300
optimizer = 'adam'
loss = 'mse'
n_layers = 9

network = models.Sequential()
network.add(layers.Dense(hidden_nodes,activation='relu',input_dim=26))
for i in range(n_layers):
    network.add(layers.Dense(hidden_nodes,activation='relu'))
network.add(layers.Dense(20,activation='linear'))
network.compile(optimizer=optimizer,loss='mse',metrics=['mae'])

network.load_weights(workDir + 'work/nbody/weights5/final_5body.h5')

# predictions = network.predict(y_test)
# for true, pred in zip(y_test,predictions):
#     for i in range(len(pred)):
#         mae += pred[i] - true[i]
#     mae /= len(pred)
#     mae_list.append(mae)

scores = network.evaluate(X_test,y_test, verbose=0)
print(network.metrics_names)

print(scores[0])
print(scores[1])