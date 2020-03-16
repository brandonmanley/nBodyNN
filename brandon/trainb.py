#Using imput data of three body motion, trains neural network to predict
#the positions of the bodies at a given time using initial positions
#By Luca Lavezzo, Brandon Manley, Jan. 2020
import pandas as pd
# from plotly.offline import iplot
# import plotly.graph_objs as go
import numpy as np
from sklearn import metrics
import math, glob
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
import keras.backend as K
import tensorflow as tf

def concatCSV(filename):
    all_filenames = [i for i in glob.glob(filename+'*.csv')]
    return pd.concat([pd.read_csv(f) for f in all_filenames], sort=False)

workDir = "/nBodyData/"
dataDir = "/nBodyData/julSim/julia_"

df = concatCSV(dataDir+'batch5')

def modified_mse(y_true,y_pred):

	# masses = y_true[:,3]
	# p0 = y_true[:,3:8]  #initial positions
	# y_true_tmp = y_true[:,10:] #final positions and velocities
	y_true_tmp = tf.slice(y_true, [0,10],[1, 12])
	# print(K.print_tensor(y_true_tmp))
	p1 = y_pred  #predicted final positions and velocities

	mse = K.mean(K.square(y_pred-y_true_tmp),axis=-1)	
	print(mse)#mean squared error between predicted and true

	#intial and final CM, delta
	# cm_x_i = (masses[0]*p0[0]+masses[1]*p0[1]+masses[2]*p0[2])/(masses[0]+masses[1]+masses[2])
	# cm_y_i = (masses[0]*p0[3]+masses[1]*p0[4]+masses[2]*p0[4])/(masses[0]+masses[1]+masses[2])
	# cm_x_f = (masses[0]*p1[0]+masses[1]*p1[1]+masses[2]*p1[2])/(masses[0]+masses[1]+masses[2])
	# cm_y_f = (masses[0]*p1[3]+masses[1]*p1[4]+masses[2]*p1[4])/(masses[0]+masses[1]+masses[2])
	# delta_cm_x = abs(cm_x_i-cm_x_f)
	# delta_cm_y = abs(cm_y_i-cm_y_f)

	return mse


def mod_acc(y_true, y_pred):
	return K.mean(K.equal(y_true[:,10:], K.round(y_pred)))


workDir = "/nBodyData/"
dataDir = "/nBodyData/julSim/julia_"

df = concatCSV(dataDir+'batch5')
# print(df.shape)

dfShuffle = shuffle(df,random_state=42)

i_col = ["m1","m2", "m3","x1", "x2", "x3", "y1", "y2", "y3", "tEnd"]
o_col = ["x1tEnd", "x2tEnd", "x3tEnd", "y1tEnd", "y2tEnd", "y3tEnd", "dx1tEnd", "dx2tEnd", "dx3tEnd", "dy1tEnd", "dy2tEnd", "dy3tEnd","eventID"]

X1 = dfShuffle[i_col].values
y1 = dfShuffle[o_col+i_col].values
# y1 = dfShuffle[o_col].values

X_train,X_test,y_train,y_test = train_test_split(X1,y1, test_size=0.2, random_state=42)

#extract id list from the y arrays
id_list_train = y_train[:,len(o_col+i_col)-1]
id_list_test = y_test[:,len(o_col+i_col)-1]
y_train = np.delete(y_train,len(o_col+i_col)-1,1)
y_test = np.delete(y_test,len(o_col+i_col)-1,1)
# id_list_train = y_train[:,len(o_col)-1]
# id_list_test = y_test[:,len(o_col)-1]
# y_train = np.delete(y_train,len(o_col)-1,1)
# y_test = np.delete(y_test,len(o_col)-1,1)

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')

print(y_train.shape,y_test.shape)

#parameters 
max_epochs = 300
optimizer = 'adam'
batch_size = 128       #FIXME: paper used 5000 for 10000 events
loss_function = "mse"   #or modified_mse

#early stopping
patienceCount = 20
callbacks = [EarlyStopping(monitor='val_loss', patience=patienceCount),
             ModelCheckpoint(filepath=workDir+'/weights/best_model.h5', monitor='val_loss', save_best_only=True)]

network = models.Sequential()
network.add(layers.Dense(128,activation='relu',input_dim=len(i_col)))
for i in range(9):
	network.add(layers.Dense(128,activation='relu'))
network.add(layers.Dense(12,activation='linear'))
network.compile(optimizer=optimizer,loss=loss_function,metrics=['accuracy'])
network.save_weights(workDir + '/weights/model_init.h5')

y_train_tmp = y_train[:,10:]
y_test_tmp =  y_test[:, 10:]
# y_test_tmp  = tf.slice(y_test, [0,0],[:, 10:])

# history = network.fit(X_train,y_train[:,10:],
#                               epochs=max_epochs,
#                               callbacks = callbacks,
#                               batch_size=batch_size,
#                               validation_data=(X_test,y_test[:,10:]),
#                               verbose = 1)

history = network.fit(X_train,y_train_tmp,
                              epochs=max_epochs,
                              callbacks = callbacks,
                              batch_size=batch_size,
                              validation_data=(X_test,y_test_tmp),
                              verbose = 1)

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

pred_out = np.asarray(predictions)
id_list_test = np.reshape(id_list_test,(id_list_test.shape[0],1))
pred_out = np.concatenate((pred_out,id_list_test),axis=1)
np.savetxt(workDir+"predicted_paths.csv", pred_out, delimiter=",")

sim_out = np.asarray(y_test)
sim_out = np.concatenate((sim_out,id_list_test),axis=1)
np.savetxt(workDir+"sim_paths.csv", pred_out, delimiter=",")