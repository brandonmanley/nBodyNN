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
dataPath = workDir + "data/nbody/4body/"
        
df_X_train = pd.read_csv(dataPath+"X_train_4body.csv")
df_X_test = pd.read_csv(dataPath+"X_test_4body.csv")
df_y_train = pd.read_csv(dataPath+"y_train_4body.csv")

X_train = df_X_train.to_numpy()
y_train = df_y_train.to_numpy()
X_test = df_X_test.to_numpy()

df1 = pd.read_csv(dataPath+"brutus10_1_4.csv")
df2 = pd.read_csv(dataPath+"brutus10_2_4.csv")
df = pd.concat([df1,df2])

test_rows = []
eps = 0.00001
X_test = X_test.astype('float64')
df = df.astype('float64')
print(df.head())
print(X_test)
print(X_test.shape)
for row in X_test:

    i = df.loc[(df['m1'] < row[0] + eps) & (df['m1'] > row[0] - eps) & (df['t'] < row[20] + eps) & (df['t'] > row[20] - eps)].index[0]
    test_rows.append(i)

dfTest = df.iloc[test_rows,:] 
o_col = []
colNames = ["x", "y", "dx", "dy"]
nBodies = 4
for col in colNames:
    for n in range(1, nBodies+1):
        o_col.append(col+"f"+str(n))

df_y_test = dfTest[o_col]
y_test = df_y_test.to_numpy()

print(X_test.shape,y_test.shape)

#Run the neural network with the best number of hidden nodes and epochs
hidden_nodes = 300   
n_epochs = 300
optimizer = 'adam'
loss = 'mse'
n_layers = 9

network = models.Sequential()
network.add(layers.Dense(hidden_nodes,activation='relu',input_dim=21))
for i in range(n_layers):
    network.add(layers.Dense(hidden_nodes,activation='relu'))
network.add(layers.Dense(16,activation='linear'))
network.compile(optimizer=optimizer,loss='mse',metrics=['mae'])
network.save_weights(workDir + 'work/nbody/weights4/model_init.h5')

history = network.fit(X_train,y_train,
                      epochs=n_epochs,
                      batch_size=500,
                      verbose=1,
                      validation_data=(X_test,y_test))

network.save_weights(workDir + 'work/nbody/weights4/final_4body.h5')

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
plt.savefig(dataPath + 'model_mae.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(dataPath + 'model_loss.png')
plt.show()


import csv
with open(dataPath+'results_final_4body.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(history.history['mae'])
    writer.writerow(history.history['val_mae'])
    writer.writerow(history.history['loss'])
    writer.writerow(history.history['val_loss'])
 