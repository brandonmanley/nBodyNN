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
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation

workDir = "/users/PAS1585/llavez99/work/nbody/"
dataDir = "/users/PAS1585/llavez99/data/nbody/"

def nested_defaultdict(default_factory, depth=1):
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()    

def kfold_network(X, y, hidden_nodes,activation='relu',optimizer='adam'):

    max_epochs = 300
    
    numSplits = 0
    
    network = models.Sequential()
    network.add(layers.Dense(hidden_nodes,activation='relu',input_dim=16))
    for i in range(9):
        network.add(layers.Dense(128,activation='relu'))
    network.add(layers.Dense(12,activation='linear'))
    network.compile(optimizer=optimizer,loss='mean_squared_logarithmic_error',metrics=['accuracy'])
    network.save_weights(workDir + '/weights/model_init.h5')
    
    #early stopping
    patienceCount = 20
    callbacks = [EarlyStopping(monitor='val_loss', patience=patienceCount),
                 ModelCheckpoint(filepath=workDir+'/weights/best_model_split'+str(numSplits)+'_nhidden'+str(hidden_nodes)+'.h5', monitor='val_loss', save_best_only=True)]

    #k-fold validation with 4 folds
    kfolds = 4
    
    training_vals_acc = 0
    training_vals_loss = 0
    valid_vals_acc = 0
    valid_vals_loss = 0
    iterations = 0
    
    avg_acc = 0
    avg_loss = 0
    avg_iterations = 0

    #k-fold validation with 5 folds
    kfolds = 5
    skf = KFold(n_splits=kfolds)

    for train_index, val_index in skf.split(X, y):

        print("Training on numSplit:",numSplits)
        numSplits += 1
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]

        network.load_weights(workDir + '/weights/model_init.h5')
        history = network.fit(X_train,y_train,
                              callbacks = callbacks,
                              epochs=max_epochs,
                              batch_size=1000,
                              validation_data=(X_val,y_val), 
                              verbose = 1)
        
        network.save(workDir + '/weights/trained_model_split'+str(numSplits)+'_nhidden'+str(hidden_nodes)+'.h5')

        plt.clf()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.savefig(workDir + '/plots/nodes'+str(hidden_nodes)+'_split'+str(numSplits)+'_accuracy.png')
        
        #save the metrics for the best epoch, or the last one
        if(len(history.history['accuracy']) == max_epochs):
            iterations += max_epochs
            training_vals_acc += history.history['accuracy'][max_epochs-1]
            training_vals_loss += history.history['loss'][max_epochs-1]
            valid_vals_acc += history.history['val_accuracy'][max_epochs-1]
            valid_vals_loss += history.history['val_loss'][max_epochs-1]
        else:
            iterations += len(history.history['accuracy']) - 10
            i = len(history.history['accuracy']) - 10 - 1
            training_vals_acc += history.history['accuracy'][i]
            training_vals_loss += history.history['loss'][i]
            valid_vals_acc += history.history['val_accuracy'][i]
            valid_vals_loss += history.history['val_loss'][i]
           
        	
    training_vals_acc /= numSplits
    training_vals_loss /= numSplits
    valid_vals_acc /= numSplits
    valid_vals_loss /= numSplits
    iterations /= numSplits*1.0

    avg_acc = valid_vals_acc
    avg_loss = valid_vals_loss
    avg_iterations = iterations
    

    # Return the average accuracy and loss and iterations (on the validation sample!)
    return avg_acc,avg_loss, avg_iterations






workDir = "/mnt/c/Users/llave/Documents/nBody/"

#Import data
fname = workDir + "data/batch_brutus10_4.csv"
df = pd.read_csv(fname)
with pd.option_context('mode.use_inf_as_null', True):
    df = df.dropna()

acc_list = []
loss_list = []
iterations_list = []
nodes_list = [128]

#dfShuffle = shuffle(df,random_state=42)
i_col = ["m1","m2","m3","x1", "x2", "x3", "y1", "y2", "y3",
    "dx1","dx2","dx3","dy1","dy2","dy3","tEnd"]
o_col = ["x1tEnd", "x2tEnd", "x3tEnd", "y1tEnd", "y2tEnd", "y3tEnd",
         "dx1tEnd", "dx2tEnd", "dx3tEnd", "dy1tEnd", "dy2tEnd", "dy3tEnd"]
X = df.as_matrix(columns=i_col)
y = df.as_matrix(columns=o_col)

X = X.astype('float64')
y = y.astype('float64')

# Determine best number of hidden nodes for one charge, and apply it for other charges
for nodes in nodes_list:
    
    print("Training:")
    print("Nodes:", nodes)
    
    #run train data through the network
    avg_acc,avg_loss,avg_iterations = kfold_network(X, y, nodes)
    
    #store and output results
    acc_list.append(avg_acc)
    loss_list.append(avg_loss)
    iterations_list.append(avg_iterations)
    
    print(avg_acc, avg_loss, avg_iterations)

plt.clf()
plt.plot(nodes_list,acc_list)
plt.ylabel('Accuracy')
plt.xlabel('Number of Hidden Nodes')
plt.savefig(workDir+"/plots/accuracy_nodes.png")

plt.clf()
plt.plot(nodes_list,loss_list)
plt.ylabel('Loss')
plt.xlabel('Number of Hidden Nodes')
plt.savefig(workDir+"/plots/loss_nodes.png")

plt.clf()
plt.plot(nodes_list,iterations_list)
plt.ylabel('Iterations (Epochs)')
plt.xlabel('Number of Hidden Nodes')
plt.savefig(workDir+"/plots/iterations_nodes.png")
