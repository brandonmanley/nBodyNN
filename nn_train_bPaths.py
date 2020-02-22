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
from keras.models import load_model
import utilityScripts.preputil as util

workDir = "/nBodyData/"
dataDir = workDir

#Import data
df = pd.read_csv(dataDir+'mathSim/batch3_1.csv')
# print(df.shape)

dfShuffle = shuffle(df,random_state=42)
# print(dfShuffle.head)

X1 = dfShuffle.as_matrix(columns=["x1", "x2", "x3", "y1", "y2", "y3", "tEnd"])
y1 = dfShuffle.as_matrix(columns=["x1tEnd", "x2tEnd", "x3tEnd", "y1tEnd", "y2tEnd", "y3tEnd","eventID"])

X_train,X_test,y_train,y_test = train_test_split(X1,y1, test_size=0.2, random_state=42)
# print(X_train.shape, y_train.shape)
# print(X_test.shape,y_test.shape)

#extract id list from the y arrays
id_list = y_test[:,6]
y_train = np.delete(y_train,6,1)
y_test = np.delete(y_test,6,1)

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')

# print(y_train.shape,y_test.shape)

#Run the neural network with the best number of hidden nodes and epochs  
n_epochs = 10
optimizer = 'adam'
loss = 'mean_squared_error'

network = models.Sequential()
network.add(layers.Dense(128,activation='relu',input_dim=7))
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
network.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
network.save_weights(workDir + '/weights/model_init.h5')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint(workDir + '/weights/best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history = network.fit(X_train,y_train, epochs=n_epochs, batch_size=128, validation_data=(X_test,y_test), verbose = 1, callbacks=[es,mc])

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

best_network = load_model(workDir + '/weights/best_model.h5')
predictions = best_network.predict(X_test)
# predictions = network.predict(X_test)

pred_out = np.asarray(predictions)
id_list = np.reshape(id_list,(id_list.shape[0],1))
pred_out = np.concatenate((pred_out,id_list),axis=1)
np.savetxt(workDir+"pred/predicted_paths_batch_3_1.csv", pred_out, delimiter=",")