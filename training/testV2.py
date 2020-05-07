import pandas as pd
from plotly.offline import iplot
import plotly.graph_objs as go
import numpy as np
from sklearn import metrics
import math, os
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from itertools import repeat
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc
from keras.utils import to_categorical
from keras import models, layers, metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation

from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_‌​parallelism_threads=40, inter_op_parallelism_threads=40)))

# import wandb
# from wandb.keras import WandbCallback
# wandb.init(project="nbody")

# import data
nBodies = 3
workDir = "/users/PAS1585/llavez99/work/nbody/nBodyNN/training/"
dataPath = "/users/PAS1585/llavez99/data/nbody/3body/"
fname = "brutus10_1_3_final.csv"
        
df = pd.read_csv(dataPath+fname)
print(df.head())

# INPUT: want mass, x/y pos, dx/dy for n bodies -> total input is n*5 + 1 (time)
# OUTPUT: want x/y, dx/dy -> total input is n*4
i_col, o_col = [], []
colNames = ["m", "x", "y", "dx", "dy"]

for col in colNames:
    for n in range(1, nBodies+1):
        i_col.append(col+str(n))
        if col != "m":
            o_col.append(col+"f"+str(n))
i_col.append("t")
    
X1 = df.as_matrix(columns=i_col)
y1 = df.as_matrix(columns=o_col)

X_train,X_test,y_train,y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

loss_fn = 'mse'
metrics = ['mae']
epochs = 500
batch_size = 1000

network = models.Sequential()
network.add(layers.Dense(X_train.shape[1], activation='relu', input_dim=X_train.shape[1]))
for i in range(10):
    network.add(layers.Dense(128,activation='relu'))
network.add(layers.Dense(y_train.shape[1],activation='linear'))
network.compile(optimizer='adam', loss=loss_fn, metrics=metrics)
network.save_weights(workDir + 'weights/model_init_test.h5')
network.load_weights(workDir + 'weights/model_init_test.h5')
history = network.fit(X_train,y_train,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(X_test,y_test), 
                      verbose = 1)

#network.save(os.path.join(wandb.run.dir, "model.h5"))

# Plot training & validation accuracy values
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model mae')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
#wandb.log({"mae": plt})

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
#wandb.log({"loss": plt})
