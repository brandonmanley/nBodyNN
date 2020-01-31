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
weightsFile = ''

#Import data
fname_test = workDir + "data/"
dfTest = pd.read_csv(fname_test)

#Train and test
X_test = dfTest.as_matrix(columns=["x1", "x2", "x3", "y1", "y2", "y3", "tEnd"])
y_test = dfTest.as_matrix(columns=["x1tEnd", "x2tEnd", "x3tEnd", "y1tEnd", "y2tEnd", "y3tEnd","eventID"])

#extract id list from the y arrays
id_list_test = y_test[:,6]
y_test = np.delete(y_test,6,1)
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
network.load_weights(workDir + weightsFile)


predictions = network.predict(X_test)

epsilon = 0.1
print("Epsilon",epsilon)
print("Precicted accurately",good_pred)
print("Predicted inaccurately",bad_pred)

pred_out = np.asarray(predictions)
id_list_test = np.reshape(id_list_test,(id_list_test.shape[0],1))
pred_out = np.concatenate((pred_out,id_list_test),axis=1)
np.savetxt(workDir + "predicted_paths.csv", pred_out, delimiter=",")