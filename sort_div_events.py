import os
import pandas as pd 
import math
import numpy as np 
import matplotlib.pyplot as plt
import random as r 
import glob
import seaborn as sns
from sklearn import metrics
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
sns.set()
pd.options.mode.chained_assignment = None  # default='warn'

simfilepath = "/mnt/c/Users/llave/Downloads/batch_brutus9_1.csv"
workDir = '/mnt/c/users/llave/Documents/nBody'
names = ["eventID","m1", "m2", "m3", 
         "x1", "x2", "x3", "y1", "y2", "y3", "dx1", "dx2", "dx3", "dy1", "dy2", "dy3",
         "tEnd",
         "x1f", "x2f", "x3f", "y1f", "y2f", "y3f", "dx1f", "dx2f", "dx3f", "dy1f", "dy2f", "dy3f"]

df = pd.read_csv(simfilepath,names=names)

inputCols = ["m1", "m2", "m3", "x1", "x2", "x3", "y1", "y2", "y3", "dx1", "dx2", "dx3", "dy1", "dy2", "dy3"]
df_new = pd.DataFrame(columns = inputCols)
n_div = 0
div_col = []

n_div = 0

for i in range(int(df.shape[0]/2560+1)):
    
    index_i = i*10000
    index_f = i*10000+2560
    event_pdf = df[(df['eventID'] >= index_i) & (df['eventID'] <= (index_f))]

    if(event_pdf.shape[0] <1): continue
    tmp_df = event_pdf[inputCols]
    df_new.loc[i]=tmp_df.iloc[0,:]

    if(event_pdf.shape[0]<2560):
        div_col.append("1")
        n_div+=1
    else:
        div_col.append("0")
print(len(div_col),df_new.shape)
df_new["div"] = div_col
print(n_div)

######################################################################
# nn stuffz
######################################################################

data = df_new.iloc[:,:15]
X1 = data.to_numpy()
y1 = df_new["div"].values 

X_train,X_test,y_train,y_test = train_test_split(X1,y1, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape,y_test.shape)

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

print("TESTING",X_train.shape)

#parameters 
max_epochs = 50
activation = 'tanh'
optimizer = 'adam'
batch_size = 20      

#
network = models.Sequential()
network.add(layers.Dense(64,activation=activation,input_shape=(15,)))
network.add(layers.Dense(2,activation='softmax'))
network.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
network.save_weights(workDir + '/weights/model_init.h5')

history = network.fit(X_train,y_train_cat,
                              epochs=max_epochs,  
                              batch_size=batch_size,
                              validation_data=(X_test,y_test_cat),
                              verbose = 0)

network.save_weights(workDir + '/weights/div_weights.h5')

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