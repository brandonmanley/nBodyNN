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

def nested_defaultdict(default_factory, depth=1):
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()

def kfold_network(X, y, hidden_nodes,activation='relu',optimizer='adam'):

    max_epochs = 500
    
    numSplits = 0
    
    network = models.Sequential()
    network.add(layers.Dense(hidden_nodes,activation='relu',input_dim=7))
    network.add(layers.Dense(6,activation='linear'))
    network.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    network.save_weights('model_init.h5')

    #early stopping
    patienceCount = 10
    callbacks = [EarlyStopping(monitor='val_loss', patience=patienceCount),
                 ModelCheckpoint(filepath='best_model_split'+str(numSplits)+'_nhidden'+str(hidden_nodes)+'.h5', monitor='val_loss', save_best_only=True)]

    #k-fold validation with 4 folds
    kfolds = 4
    skf = StratifiedKFold(n_splits=kfolds)
    
    training_vals_acc = 0
    training_vals_loss = 0
    valid_vals_acc = 0
    valid_vals_loss = 0
    iterations = 0
    
    avg_acc = 0
    avg_loss = 0
    avg_iterations = 0

    X_split = numpy.array_split(numpy.array(X),numSplits)
    y_split = numpy.array_split(numpy.array(y),numSplits)

    for index in range(numSplits):
        
        print("Training: numSplits", numSplits)
        numSplits += 1

        X_train_temp = []
        y_train_temp = []
        X_val = []
        y_val = []
                
        for jindex in range(numSplits):
            if index == jindex:
                X_val = X[jindex]
                y_val = y[jindex]
            else:
                X_train_temp.append(X_split[index])
                y_train_temp.append(y_split[index])
        
        

        network.load_weights('model_init.h5')
        history = network.fit(X_train_temp,y_train_temp,
                              callbacks = callbacks,
                              epochs=max_epochs,
                              batch_size=128,
                              validation_data=(X_val,y_val), 
                              verbose = 0)
        
        network.save('trained_model_split'+str(numSplits)+'_nhidden'+str(hidden_nodes)+'.h5')
    
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
fname = "/mnt/c/users/llave/Downloads/valData_500_t7.csv"
df = pd.read_csv(fname)
print(df.head)

dfShuffle = shuffle(df,random_state=42)
X1 = dfShuffle.as_matrix(columns=["x1", "x2", "x3", "y1", "y2", "y3", "tEnd"])
y1 = dfShuffle.as_matrix(columns=["x1[tEnd]", "x2[tEnd]", "x3[tEnd]", "y1[tEnd]", "y2[tEnd]", "y3[tEnd]"])

X_train,X_test,y_train,y_test = train_test_split(X1,y1, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape,y_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

acc_list = []
loss_list = []
iterations_list = []

# Determine best number of hidden nodes for one charge, and apply it for other charges
for nodes in [10,20,30,50,100]:
    
    print("Nodes", nodes)
    
    #run train data through the network
    avg_acc,avg_loss,avg_iterations = kfold_network(X_train, y_train, nodes)
    
    #store and output results
    acc_list.append(avg_acc)
    loss_list.append(avg_loss)
    iterations_list.append(avg_iterations)
    
    print(avg_acc, avg_loss, avg_iterations)

plt.plot([10,20,30,50,100],acc_list)
plt.ylabel('Accuracy')
plt.xlabel('Number of Hidden Nodes')

plt.plot([10,20,30,50,100],loss_list)

plt.plot([10,20,30,50,100],iterations_list)