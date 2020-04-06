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
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation

workDir = "/users/PAS1585/llavez99/"

# returns dataframe
# params: full -> True = full dataset, False = first file

def grab_data(full, cols, path):
    if full:
        df = pd.DataFrame()
        for file in os.listdir(path):
            if ".csv" not in file: continue
            if "train" or "test" in file: continue
            dftemp =  pd.read_csv(path+file, index_col=False) 
            df = pd.concat([df,dftemp])
    else:
        for file in os.listdir(path):
            if "10_4" not in file: continue
            return pd.read_csv(path+file, names=cols, index_col=False)


#Import data
dataPath = workDir + "data/nbody/4body/"
nBodies = 4
dataCols = ["file", "eventID"]
perParticleColumnsInput = ["m", "x", "y", "dx", "dy"]
perParticleColumnsOutput = ["xf", "yf", "dxf", "dyf"]

for col in perParticleColumnsInput:
    for i in range(nBodies):
        dataCols.append(col+str(i+1))
dataCols.append("t")
for col in perParticleColumnsOutput:
    for i in range(nBodies):
        dataCols.append(col+str(i+1))
        
df = grab_data(True, dataCols, dataPath)

with pd.option_context('mode.use_inf_as_null', True):
    #df = df.dropna(axis=1)
    df = df.dropna(axis=0)
    
dfShuffle = shuffle(df,random_state=42)

i_col, o_col = [], []
colNames = ["m", "x", "y", "dx", "dy"]

for col in colNames:
    for n in range(1, nBodies+1):
        i_col.append(col+str(n))
        if col != "m":
            o_col.append(col+"f"+str(n))
i_col.append("t")
    
X1 = df.as_matrix(columns=i_col)
y1 = df.as_matrix(columns=o_col)

X1 = X1.astype('float64')
y1 = y1.astype('float64')

X_train,X_test,y_train,y_test = train_test_split(X1,y1, test_size=0.01, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape,y_test.shape)
np.savetxt(workDir+"data/nbody/4body/X_train_4body.csv", X_train, delimiter=",")
np.savetxt(workDir+"data/nbody/4body/y_train_4body.csv", y_train, delimiter=",")
np.savetxt(workDir+"data/nbody/4body/X_test_4body.csv", X_test, delimiter=",")
np.savetxt(workDir+"data/nbody/4body/y_test_4body.csv", X_test, delimiter=",")

def nested_defaultdict(default_factory, depth=1):
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()    

def kfold_network(X, y, hidden_nodes,n_layers,activation='relu',optimizer='adam'):

    max_epochs = 300
    
    numSplits = 0
    
    network = models.Sequential()
    network.add(layers.Dense(hidden_nodes,activation='relu',input_dim=16))
    for i in range(n_layers):
        network.add(layers.Dense(hidden_nodes,activation='relu'))
    network.add(layers.Dense(12,activation='linear'))
    network.compile(optimizer=optimizer,loss='mse',metrics=['mae'])
    network.save_weights(workDir + 'work/nbody/weights/model_init.h5')
    
    #early stopping
    patienceCount = 30
    callbacks = [EarlyStopping(monitor='val_loss', patience=patienceCount),
                 ModelCheckpoint(filepath=workDir+'work/nbody/weights/best_model_split'+str(numSplits)+'_nlayers'+str(n_layers)+'_nhidden'+str(hidden_nodes)+'.h5', monitor='val_loss', save_best_only=True)]

    training_vals_mae = 0
    training_vals_loss = 0
    valid_vals_mae = 0
    valid_vals_loss = 0
    iterations = 0
    
    avg_mae = 0
    avg_loss = 0
    avg_iterations = 0

    #k-fold validation with 5 folds
    kfolds = 4
    skf = KFold(n_splits=kfolds)

    for train_index, val_index in skf.split(X, y):

        print("Training on numSplit:",numSplits)
        numSplits += 1
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]

        network.load_weights(workDir + 'work/nbdoy/weights/model_init.h5')
        history = network.fit(X_train,y_train,
                              callbacks = callbacks,
                              epochs=max_epochs,
                              batch_size=1000,
                              validation_data=(X_val,y_val), 
                              verbose = 1)
        
        network.save(workDir + 'work/nbody/weights/trained_model_split'+str(numSplits)+'_nhidden'+str(hidden_nodes)+'.h5')

        plt.clf()
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.savefig(workDir + 'work/nbody/plots/layers'+str(n_layers)+'_nodes'+str(hidden_nodes)+'_split'+str(numSplits)+'_mae.png')
        
        #save the metrics for the best epoch, or the last one
        if(len(history.history['mae']) == max_epochs):
            iterations += max_epochs
            training_vals_mae += history.history['mae'][max_epochs-1]
            training_vals_loss += history.history['loss'][max_epochs-1]
            valid_vals_mae += history.history['val_mae'][max_epochs-1]
            valid_vals_loss += history.history['val_loss'][max_epochs-1]
        else:
            iterations += len(history.history['mae']) - 10
            i = len(history.history['mae']) - 10 - 1
            training_vals_mae += history.history['mae'][i]
            training_vals_loss += history.history['loss'][i]
            valid_vals_mae += history.history['val_mae'][i]
            valid_vals_loss += history.history['val_loss'][i]
           
        	
    training_vals_mae /= numSplits
    training_vals_loss /= numSplits
    valid_vals_mae /= numSplits
    valid_vals_loss /= numSplits
    iterations /= numSplits*1.0

    avg_mae = valid_vals_mae
    avg_loss = valid_vals_loss
    avg_iterations = iterations
    

    # Return the average MAE and loss and iterations (on the validation sample!)
    return avg_mae,avg_loss, avg_iterations



parameters_list = []
mae_list = []
loss_list = []
iterations_list = []
nodes_list = [50,128,200,300]
layers_list = [1,5,10,15]

# Determine best number of hidden nodes for one charge, and apply it for other charges
for iLayer in layers_list:
    for iNode in nodes_list:
    
        print("Training:")
        print("Layers:",iLayer)
        print("Nodes:", iNode)
        
        #run train data through the network
        avg_mae,avg_loss,avg_iterations = kfold_network(X_train, y_train, iNode,iLayer)
        
        #store and output results
        parameters_list.append([iLayer,iNode])
        mae_list.append(avg_mae)
        loss_list.append(avg_loss)
        iterations_list.append(avg_iterations)
        
        print(avg_mae, avg_loss, avg_iterations)

print(parameters_list)
print(mae_list)
print(loss_list)
print(iterations_list)