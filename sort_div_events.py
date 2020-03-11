import os
import pandas as pd 
import math
import numpy as np 
import matplotlib.pyplot as plt
import random as r 
import glob
import seaborn as sns
sns.set()
pd.options.mode.chained_assignment = None  # default='warn'

simfilepath = "/mnt/c/Users/llave/Downloads/batch_brutus9_1.csv"
df = pd.read_csv(simfilepath)

inputCols = ["m1", "m2", "m3", "x1", "x2", "x3", "y1", "y2", "y3", "dx1", "dx2", "dx3", "dy1", "dy2", "dy3"]
df_new = pd.DataFrame()
n_div = 0
div_col = []
for i in range(280):
    index_i = i*10000
    index_f = i*10000+2560
    event_pdf = df.loc[(df['eventID'] >= index_i) & (df['eventID'] <= (index_f))]
    
    tmp_df = df[inputCols]
    df_new = pd.concat([tmp_df.iloc[[0]], df_new])

    if(event_pdf.shape[0]<2560):
        div_col.append("1")
    else:
        div_col.append("0")

df_new["div"] = div_col

print(n_div)
print(df_div.shape)



######################################################################
# nn stuffz
######################################################################

data = df_new[:,15]
X1 = data.values
y1 = df_new["div"].values

X_train,X_test,y_train,y_test = train_test_split(X1,y1, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape,y_test.shape)

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')

print(y_train.shape,y_test.shape)

#parameters 
max_epochs = 100
optimizer = 'adam'
batch_size = 20      
loss_function = 'mse'   

#early stopping
patienceCount = 20
callbacks = [EarlyStopping(monitor='val_loss', patience=patienceCount),
             ModelCheckpoint(filepath=workDir+'/weights/best_model.h5', monitor='val_loss', save_best_only=True)]

network = models.Sequential()
network.add(layers.Dense(64,activation='relu',input_dim=15))
network.add(layers.Dense(1,activation='softmax'))
network.compile(optimizer=optimizer,loss=loss_function,metrics=['accuracy'])
network.save_weights(workDir + '/weights/model_init.h5')

history = network.fit(X_train,y_train,
                              epochs=max_epochs,
                              callbacks = callbacks,
                              batch_size=batch_size,
                              validation_data=(X_test,y_test),
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