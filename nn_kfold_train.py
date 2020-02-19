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
import preputil as util

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
    network.add(layers.Dense(hidden_nodes,activation='relu',input_dim=7))
    network.add(layers.Dense(128,activation='relu'))
    network.add(layers.Dense(128,activation='relu'))
    network.add(layers.Dense(128,activation='relu'))
    network.add(layers.Dense(128,activation='relu'))
    network.add(layers.Dense(128,activation='relu'))
    network.add(layers.Dense(128,activation='relu'))
    network.add(layers.Dense(128,activation='relu'))
    network.add(layers.Dense(128,activation='relu'))
    network.add(layers.Dense(128,activation='relu'))
    network.add(layers.Dense(6,activation='linear'))
    network.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['accuracy'])
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

    numEvents = X.shape[0]
    numEventsSplit = int(numEvents/4)
    indeces = [numEventsSplit, numEventsSplit*2,numEventsSplit*3]

    #Testing
    print(X.shape[0], indeces)

    X_s1=X[:indeces[0],:]
    X_s2=X[indeces[0]:indeces[1],:]
    X_s3=X[indeces[1]:indeces[2],:]
    X_s4=X[indeces[2]:,:]
    y_s1=y[:indeces[0],:]
    y_s2=y[indeces[0]:indeces[1],:]
    y_s3=y[indeces[1]:indeces[2],:]
    y_s4=y[indeces[2]:,:]
    X_split = [X_s1,X_s2,X_s3,X_s4]
    y_split = [y_s1,y_s2,y_s3,y_s4]

    for index in range(kfolds):
        
        print("Training: numSplits", numSplits)
        numSplits += 1

        X_train_temp = []
        y_train_temp = []
        X_val = []
        y_val = []
        i = []
                
        for jindex in range(kfolds):
            if index == jindex:
                X_val = X_split[jindex]
                y_val = y_split[jindex]
            else:
                i.append(jindex)
        
        X_train_temp = np.vstack((X_split[i[0]],X_split[i[1]],X_split[i[2]]))
        y_train_temp = np.vstack((y_split[i[0]],y_split[i[1]],y_split[i[2]]))

        network.load_weights(workDir + '/weights/model_init.h5')
        history = network.fit(X_train_temp,y_train_temp,
                              callbacks = callbacks,
                              epochs=max_epochs,
                              batch_size=128,
                              validation_data=(X_val,y_val), 
                              verbose = 0)
        
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







#Import data
df = util.concatCSV(dataDir+'batch3')
print(df.shape)

acc_list = []
loss_list = []
iterations_list = []
nodes_list = [128]

# Determine best number of hidden nodes for one charge, and apply it for other charges
for nodes in nodes_list:

    dfShuffle = shuffle(df,random_state=42)
    X = dfShuffle.as_matrix(columns=["x1", "x2", "x3", "y1", "y2", "y3", "tEnd"])
    y = dfShuffle.as_matrix(columns=["x1tEnd", "x2tEnd", "x3tEnd", "y1tEnd", "y2tEnd", "y3tEnd"])

    X = X.astype('float64')
    y = y.astype('float64')
    
    print("Nodes", nodes)
    
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
