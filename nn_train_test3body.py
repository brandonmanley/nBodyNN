#Using imput data of three body motion, trains neural network to predict
#the positions of the bodies at a given time using initial positions
#By Luca Lavezzo, Brandon Manley, Jan. 2020

import pandas as pd
from plotly.offline import iplot
import plotly.graph_objs as go
import numpy as np
import os
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
dataDir = workDir + "data/nbody/3body/"

def grab_data(full, cols, path):
    if full:
        df = pd.DataFrame()
        for file in os.listdir(path):
            if "final" not in file: continue
            if "brutus10" not in file: continue
            if "train" in file or "test" in file: continue
            dftemp =  pd.read_csv(path+file, index_col=False) 
            df = pd.concat([df,dftemp])
        return df
    else:
        for file in os.listdir(path):
            if "10_4" not in file: continue
            return pd.read_csv(path+file, names=cols, index_col=False)

    
dfTrainIndeces = pd.read_csv(dataDir+"trainMap3body.csv")
dfTestIndeces = pd.read_csv(dataDir+"testMap3body.csv")

dfTrainIndeces = dfTrainIndeces.astype(int)
dfTestIndeces = dfTestIndeces.astype(int)

df = grab_data(True, [], dataDir)

df['eventID'] = df['eventID'].astype(int)
df['finalFile'] = df['finalFile'].astype(int)

dfTrain = df.loc[dfTrainIndeces['eventID','finalFile']==df['eventID','finalFile']]
dfTest = df.loc[dfTestIndeces['eventID','finalFile']==df['eventID','finalFile']]

i_col, o_col = [], []
colNames = ["m", "x", "y", "dx", "dy"]
nBodies = 3
for col in colNames:
    for n in range(1, nBodies+1):
        i_col.append(col+str(n))
        if col != "m":
            o_col.append(col+"f"+str(n))
i_col.append("t")
    
X1 = df.as_matrix(columns=i_col)
y1 = df.as_matrix(columns=o_col)

X = X.astype('float32')
y = y.astype('float32')
X_test = X_test.astype('float32')
y_test = y_test.astype('float32')


#Run the neural network with the best number of hidden nodes and epochs
hidden_nodes = 128   
n_epochs = 300
optimizer = 'adam'
loss = 'mse'
n_layers = 9

etwork = models.Sequential()
network.add(layers.Dense(hidden_nodes,activation='relu',input_dim=16))
for i in range(n_layers):
    network.add(layers.Dense(hidden_nodes,activation='relu'))
network.add(layers.Dense(12,activation='linear'))
network.compile(optimizer=optimizer,loss='mse',metrics=['mae'])
network.save_weights(workDir + 'work/nbody/weights3/model_init.h5')

history = network.fit(X_train,y_train,
                      epochs=n_epochs,
                      batch_size=5000,
                      verbose=1,
                      validation_data=(X_test,y_test))

network.save_weights(workDir + 'work/nbody/weights3/final_3body.h5')

training_vals_acc = history.history['mae']
training_vals_loss = history.history['loss']
valid_vals_acc = history.history['val_mae']
valid_vals_loss = history.history['val_loss']
iterations = len(training_vals_acc)
print("Number of iterations:",iterations)
print("Epoch\t Train Loss\t Train Acc\t Val Loss\t Val Acc")
i = 0
for tl,ta,vl,va in zip(training_vals_loss,training_vals_acc,valid_vals_loss,valid_vals_acc):
    print(i,'\t',round(tl,5),'\t',round(ta,5),'\t',round(vl,5),'\t',round(va,5))
    i += 1

# Plot training & validation mae values
print(history.history.keys())
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model mae')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(dataDir + 'model_mae.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(dataDir + 'model_loss.png')
plt.show()


import csv
with open(dataDir+'results_final_3body.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(history.history['mae'])
    writer.writerow(history.history['val_mae'])
    writer.writerow(history.history['loss'])
    writer.writerow(history.history['val_loss'])
 