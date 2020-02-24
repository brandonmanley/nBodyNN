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
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Dropout, Activation
import preputil as util
import keras.backend as K

m1 = 150
m2 = 120
m3 = 130
def mse_energy_loss(i_layer):
	def loss(y_true, y_pred):
		p0=i_layer
		p1=y_pred
		energy_i = p0[0]
		energy_f = p1[0]
		mse = K.mean(K.square(y_pred-y_true),axis=-1)
		delta_energy = abs(energy_i-energy_f)
		total_loss = mse + delta_energy
		return total_loss
	return loss

def test_loss(initial):
	def loss(y_true,y_pred):
		energy_i = initial[0]
		energy_f = y_true[0]
		delta_energy = 0
		#delta_energy = energy_f-energy_i
		mse = K.mean(K.square(y_pred-y_true),axis=-1)
		total_loss = mse + delta_energy
		return total_loss
	return loss

workDir = "/users/PAS1585/llavez99/work/nbody/"
dataDir = "/users/PAS1585/llavez99/data/nbody/"

#Import data
df = util.concatCSV(dataDir+'batch3')
print(df.shape)

dfShuffle = shuffle(df,random_state=42)
print(dfShuffle.head)

X1 = dfShuffle.as_matrix(columns=["x1", "x2", "x3", "y1", "y2", "y3", "tEnd"])
y1 = dfShuffle.as_matrix(columns=["x1tEnd", "x2tEnd", "x3tEnd", "y1tEnd", "y2tEnd", "y3tEnd","eventID"])

X_train,X_test,y_train,y_test = train_test_split(X1,y1, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape,y_test.shape)

#extract id list from the y arrays
id_list_train = y_train[:,6]
id_list_test = y_test[:,6]
y_train = np.delete(y_train,6,1)
y_test = np.delete(y_test,6,1)

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')

print(y_train.shape,y_test.shape)

#Run the neural network with the best number of hidden nodes and epochs  
n_epochs = 300
optimizer = 'adam'

input_l = Input(shape=(7,))
x1 = Dense(128,activation='relu')(input_l)
x2 = Dense(128,activation='relu')(x1)
x3 = Dense(128,activation='relu')(x2)
x4 = Dense(128,activation='relu')(x3)
x5 = Dense(128,activation='relu')(x4)
x6 = Dense(128,activation='relu')(x5)
x7 = Dense(128,activation='relu')(x6)
x8 = Dense(128,activation='relu')(x7)
x9 = Dense(128,activation='relu')(x8)
x10 = Dense(128,activation='relu')(x9)
output_l = Dense(6,activation='linear')(x10)
model = Model(input_l,output_l)
model.compile(optimizer=optimizer,loss=test_loss(input_l),metrics=['accuracy'])
model.save_weights(workDir + '/weights/model_init.h5')

history = model.fit(X_train,y_train,
                              epochs=300,
                              batch_size=128,
                              validation_data=(X_test,y_test),
                              verbose = 0)

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


predictions = model.predict(X_test)

pred_out = np.asarray(predictions)
id_list_test = np.reshape(id_list_test,(id_list_test.shape[0],1))
pred_out = np.concatenate((pred_out,id_list_test),axis=1)
np.savetxt(workDir+"predicted_paths.csv", pred_out, delimiter=",")

sim_out = np.asarray(y_test)
sim_out = np.concatenate((sim_out,id_list_test),axis=1)
np.savetxt(workDir+"sim_paths.csv", pred_out, delimiter=",")