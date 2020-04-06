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
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

workDir = "/mnt/c/Users/llave/Documents/nBody/"

#Import data
fname = workDir + "data/batch_brutus10"
numFiles = 5
df = pd.DataFrame()

for i in range(numFiles):
	if i!=4: continue
	dftemp = pd.DataFrame() 
	dftemp = pd.read_csv(fname+"_"+str(i)+".csv")
	df = pd.concat([df, dftemp])

with pd.option_context('mode.use_inf_as_null', True):
    df = df.dropna()

dfShuffle = shuffle(df,random_state=42)

i_col = ["m1","m2","m3","x1", "x2", "x3", "y1", "y2", "y3",
		"dx1","dx2","dx3","dy1","dy2","dy3","tEnd"]
o_col = ["x1tEnd", "x2tEnd", "x3tEnd", "y1tEnd", "y2tEnd", "y3tEnd",
		"dx1tEnd", "dx2tEnd", "dx3tEnd", "dy1tEnd", "dy2tEnd", "dy3tEnd"]

X1 = df.as_matrix(columns=i_col)
y1 = df.as_matrix(columns=o_col)

X1 = X1.astype('float64')
y1 = y1.astype('float64')

X_train,X_test,y_train,y_test = train_test_split(X1,y1, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape,y_test.shape)


def create_model(n_nodes=128,n_layers=10):
    
    network = models.Sequential()
    network.add(layers.Dense(n_nodes,activation='relu',input_dim=16))
    for i in range(n_layers):
        network.add(layers.Dense(n_nodes,activation='relu'))
    network.add(layers.Dense(12,activation='linear'))
    network.compile(optimizer='adam',loss='mae',metrics=['mae'])
    return network

# Definying grid parameters
testing_layers = [1, 4, 6, 9, 14]
testing_nodes = [10,50,128,200]
param_grid = dict(n_layers = testing_layers, n_nodes = testing_nodes)

clf = KerasClassifier(build_fn= create_model, epochs=300, batch_size=1000, verbose= 0)

gridmodel = GridSearchCV(estimator= clf, param_grid=param_grid, cv=4,n_jobs=-1)
result = gridmodel.fit(X_train,y_train)

# print results
print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f'mean={mean:.4}, std={stdev:.4} using {param}')