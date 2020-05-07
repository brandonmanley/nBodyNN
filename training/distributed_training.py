import tensorflow as tf
import pandas as pd
import os
from sklearn.model_selection import train_test_split

print(tf.__version__)

from tensorflow.python.client import device_lib

print()
print("#################################################")
print(device_lib.list_local_devices())
print("#################################################")
print()


# returns dataframe
# params: full -> True = full dataset, False = first file

def grab_data(full, path):
    if full:
        return pd.concat([pd.read_csv(path+file, index_col=False) for file in os.listdir(path)])
    else:
        for file in os.listdir(path):
            return pd.read_csv(path+file, index_col=False)
        
# import data

nBodies = 3
workDir = "/users/PAS1585/llavez99/work/nbody/nBodyNN/training/"
dataPath = "/users/PAS1585/llavez99/data/nbody/3body/"
fname = "brutus10_1_3_final.csv"
        
df = pd.read_csv(dataPath+fname, index_col=False)
df.head()

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
    
X1 = df[i_col].values
y1 = df[o_col].values

X_train,X_test,y_train,y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

print()
print("#################################################")
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
print("#################################################")
print()

# You can also do info.splits.total_num_examples to get the total
# number of examples in the dataset.

num_train_examples = X_train.shape[0]
num_test_examples = X_test.shape[0]

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(X_train.shape[1], activation='relu', input_dim=X_train.shape[1]),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(y_train.shape[1],activation='linear')
    ])

    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['mse'])
    
# Define the checkpoint directory to store the checkpoints

checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# Function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))
        
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]

model.fit(train_ds, epochs=5, callbacks=callbacks)