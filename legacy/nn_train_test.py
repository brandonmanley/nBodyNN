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

workDir = "/mnt/c/Users/llave/Documents/nBody/"

#Import data
fname_train = workDir + "data/"
fname_test = workDir + "data/"
dfTrain = pd.read_csv(fname_train)
dfTest = pd.read_csv(fname_test)

#Shuffle
dfShuffle_train = shuffle(dfTrain,random_state=42)
dfShuffle_test = shuffle(dfTest,random_state=42)

#Train and test
X = dfShuffle_train.as_matrix(columns=["x1", "x2", "x3", "y1", "y2", "y3", "tEnd"])
y = dfShuffle_train.as_matrix(columns=["x1tEnd", "x2tEnd", "x3tEnd", "y1tEnd", "y2tEnd", "y3tEnd","eventID"])
X_test = dfShuffle_test.as_matrix(columns=["x1", "x2", "x3", "y1", "y2", "y3", "tEnd"])
y_test = dfShuffle_test.as_matrix(columns=["x1tEnd", "x2tEnd", "x3tEnd", "y1tEnd", "y2tEnd", "y3tEnd","eventID"])

#extract id list from the y arrays
id_list_test = y_test[:,6]
y_test = np.delete(y_test,6,1)

X = X.astype('float32')
y = y.astype('float32')
X_test = X_test.astype('float32')
y_test = y_test.astype('float32')


#Run the neural network with the best number of hidden nodes and epochs
hidden_nodes = 50   
n_epochs = 200
optimizer = 'adam'
loss = 'mean_squared_logarithmic_error'

network = models.Sequential()
network.add(layers.Dense(hidden_nodes,activation='relu',input_dim=7))
network.add(layers.Dense(6,activation='linear'))
network.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
network.save_weights(workDir + '/weights/model_init.h5')

history = network.fit(X,y,
                      epochs=n_epochs,
                      batch_size=128,
                      verbose=1,
                      validation_data=(X_test,y_test))

training_vals_acc = history.history['accuracy']
training_vals_loss = history.history['loss']
valid_vals_acc = history.history['val_accuracy']
valid_vals_loss = history.history['val_loss']
iterations = len(training_vals_acc)
print("Number of iterations:",iterations)
print("Epoch\t Train Loss\t Train Acc\t Val Loss\t Val Acc")
i = 0
for tl,ta,vl,va in zip(training_vals_loss,training_vals_acc,valid_vals_loss,valid_vals_acc):
    print(i,'\t',round(tl,5),'\t',round(ta,5),'\t',round(vl,5),'\t',round(va,5))
    i += 1

# Plot training & validation accuracy values
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(workDir + 'model_accuracy.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(workDir + 'model_loss.png')
plt.show()


predictions = network.predict(X_test)

epsilon = 0.1
print("Epsilon",epsilon)
print("Precicted accurately",good_pred)
print("Predicted inaccurately",bad_pred)

pred_out = np.asarray(predictions)
id_list_test = np.reshape(id_list_test,(id_list_test.shape[0],1))
pred_out = np.concatenate((pred_out,id_list_test),axis=1)
np.savetxt(workDir + "predicted_paths.csv", pred_out, delimiter=",")