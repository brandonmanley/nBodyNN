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
path = "~/Desktop/nBodyNN/"
meta_input = "1015_2020-01-26"
#Import data
fname = path+"val_"+meta_input+".csv"
df = pd.read_csv(fname)
print(df.head)

#Split variables and signal info, train and test
dfShuffle = shuffle(df,random_state=42)
dfShuffle = dfShuffle[dfShuffle.tEnd!=0]
X1 = dfShuffle.as_matrix(columns=["x1", "x2", "x3", "y1", "y2", "y3", "tEnd"])
y1 = dfShuffle.as_matrix(columns=["x1[tEnd]", "x2[tEnd]", "x3[tEnd]", "y1[tEnd]", "y2[tEnd]", "y3[tEnd]","eventID"])

X_train,X_test,y_train,y_test = train_test_split(X1,y1, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape,y_test.shape)

#extract id list from the y arrays
id_list = y_test[:,6]
y_train = np.delete(y_train,6,1)
y_test = np.delete(y_test,6,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
print(y_train.shape,y_test.shape)

#FIXME: add kfold validation to optimize nodes, epochs, layers

#Run the neural network with the best number of hidden nodes and epochs
hidden_nodes = 50   
n_epochs = 200
optimizer = 'adam'

network = models.Sequential()
network.add(layers.Dense(hidden_nodes,activation='relu',input_dim=7))
network.add(layers.Dense(6,activation='linear'))
network.compile(optimizer=optimizer,loss='mean_squared_logarithmic_error',metrics=['accuracy'])
network.save_weights('model_init.h5')

history = network.fit(X_train,y_train,
                              epochs=n_epochs,
                              batch_size=128,
                              verbose=1,
                              validation_data=(X_test,y_test))
                              

# training_vals_acc = history.history['accuracy']
# training_vals_loss = history.history['loss']
# valid_vals_acc = history.history['val_acc']
# valid_vals_loss = history.history['val_loss']
# iterations = len(training_vals_acc)
# print("Number of iterations:",iterations)
# print("Epoch\t Train Loss\t Train Acc\t Val Loss\t Val Acc")
# i = 0
# for tl,ta,vl,va in zip(training_vals_loss,training_vals_acc,valid_vals_loss,valid_vals_acc):
#     print(i,'\t',round(tl,5),'\t',round(ta,5),'\t',round(vl,5),'\t',round(va,5))
#     i += 1

# Plot training & validation accuracy values
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('model_accuracy.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('model_loss.png')
plt.show()


predictions = network.predict(X_test)
epsilon = 0.1
good_pred=0
bad_pred=0

i=0
for pred, true in zip(predictions, y_test):
  for pred_coord, true_coord in zip(pred, true):
    if abs(pred_coord-true_coord) > epsilon:
      bad_pred+=1
    else:
      good_pred+=1
  i+=1
  print(pred,true)
  if(i>10): break

print("Epsilon",epsilon)
print("Precicted accurately",good_pred)
print("Predicted inaccurately",bad_pred)

pred_out = np.asarray(predictions)
id_list = np.reshape(id_list,(id_list.shape[0],1))
pred_out = np.concatenate((pred_out,id_list),axis=1)
np.savetxt("predicted_paths_"+meta_input+".csv", pred_out, delimiter=",")