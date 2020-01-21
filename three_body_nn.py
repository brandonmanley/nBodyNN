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

#Import data
fname = "INPUT DATA"
df = pd.read_csv(fname)
df.columns = ['m1', 'm2', 'm3', 't_i','t_f',
'x1_i','x2_i','x3_i','y1_i','y2_i','y3_i',
'vx1_i','vx2_i','vx3_i','vy1_i','vy2_i','vy3_i',
'x1_f','x2_f','x3_f','y1_f','y2_f','y3_f',
'vx1_f','vx2_f','vx3_f','vy1_f','vy2_f','vy3_f',
]

#Split variables and signal info, train and test
dfShuffle = shuffle(df,random_state=42)
X1 = dfShuffle.as_matrix(columns=['COLUMNS'])
y1 = dfShuffle['COLUMNS'].values

X_train,X_test,y_train,y_test = train_test_split(X1,y1, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape,y_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

#FIXME: add kfold validation to optimize nodes, epochs, layers

#Run the neural network with the best number of hidden nodes and epochs
hidden_nodes = 10
n_epochs = 10
optimizer = 'adam'

network = models.Sequential()
network.add(layers.Dense(hidden_nodes,activation='relu',input_shape=(7,)))
network.add(layers.Dense(6,activation='linear'))
network.compile(optimizer=optimizer,loss='mean_squared_error')
network.save_weights('model_init.h5')

callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
network.load_weights('model_init.h5')

history = network.fit(X_train,y_train,
                              epochs=n_epochs,
                              batch_size=128,
                              verbose=1,
                              validation_data=(X_test,X_test))

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

network.load_weights('best_model.h5')
predictions = network.predict(X_test)
epsilon = 0.1
good_pred=0
bad_pred=0

for pred, true in zip(predicitons, y_test):
	for pred_coord, true_coord in zip(pred, true):
		if abs(pred_coord-true_coord) > epsilon:
			bad_pred+=1
		else:
			good_pred+=1

print("Epsilon",epsilon)
print("Precicted accurately",good_pred)
print("Predicted inaccurately",bad_pred)