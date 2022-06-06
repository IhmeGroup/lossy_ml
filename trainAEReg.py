#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import time
import numpy as np  
import os
import sys
import scipy.io 
from sklearn.utils import shuffle
from scipy import ndimage
import random
import argparse
import matplotlib.pyplot as plt


# In[22]:


parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=bool,default=False, help="Enable GPU logging.")
parser.add_argument("--alpha", type=float,default=1e-4, help="Raw learning rate.")
parser.add_argument("--nbatch", type=int,default=36, help="Batch size.")
parser.add_argument("--epochs", type=int,default=150, help="Epochs.")
parser.add_argument("--val_split",type=float,default=0.1, help="Val/Train Ratio.")
parser.add_argument("--train_dir", type=str,default='../data_reg/', help="Train directory.")
parser.add_argument("--lossy_e", type=str,default='', help="Train with lossy error? Format 0p1 = 0.1")
parser.add_argument("--maxSRC_H2O", type=float,default=13000, help="Max for Normalization.")
parser.add_argument("--minSRC_H2O", type=float,default=-52, help="Min for Normalization.")
parser.add_argument("--maxH2O", type=float,default=0.186137, help="Max for Normalization.")
parser.add_argument("--maxZ", type=float,default=1.0095, help="Max for Normalization.")
parser.add_argument("--firstC", type=int,default=8, help="First number of Filters/Cout.")


# In[23]:




#parse args, does not work in jupyter
args = parser.parse_args()
batch_size = args.nbatch
alpha= args.alpha *(batch_size/10.0)**0.5
epochs = args.epochs
train_dir = args.train_dir
debug = args.debug
val_split = args.val_split
maxSRC_H2O=args.maxSRC_H2O
minSRC_H2O=args.minSRC_H2O
maxH2O=args.maxH2O
maxZ=args.maxZ
firstC = args.firstC
lossy_e = args.lossy_e


# In[4]:


# X_name = ['Y H2', 'Y O2', 'Y H2O', 'ZBilger']
my_max = np.array([maxH2O,maxZ]).reshape(1,1,1,2)


# In[5]:


tf.debugging.set_log_device_placement(debug)


# In[34]:


#utils
def get_filepath_list(train_dir):
    filenames = []
    for subdir, dirs, files in os.walk(train_dir):
        #print(files)
        files.sort()
        for file in files:
            filenames.append(train_dir+file)
    return filenames

def random_rotation(my_dict):
    angles = [90, 180, 270, 360]
    angle = random.choice(angles)
    X = ndimage.rotate(my_dict["X"], angle, axes=(1, 2), reshape=False)
    y = ndimage.rotate(my_dict["y"], angle, axes=(1, 2), reshape=False)
    
    return {"X":X,"y":y}

def random_flip(my_dict):
    bools = [True,False]
    boolud = random.choice(bools)
    if boolud:
        X = my_dict["X"][:, ::-1, :]
        y = my_dict["y"][:, ::-1, :]
    else:
        X = my_dict["X"]
        y = my_dict["y"]
    return {"X":X,"y":y}

def scaleX(image,my_max=my_max):
    return np.array(image/my_max).astype(np.float32)
def scaleY(image,my_max=maxSRC_H2O,my_min = minSRC_H2O):
    return np.array((image-my_min)/(my_max-my_min)).astype(np.float32)
            
def build_autoencoder(input_shape=(3,32,32, 2),C_chan = firstC,C_mult = [2, 2, 1]):
    #encoder
    encoder_input = tf.keras.Input(shape=input_shape, name="original_solut")
    C = C_chan
    x = tf.keras.layers.Conv3D(C, 3, activation="relu",padding="same",name='conv_1')(encoder_input)
    skip_connection = x
    for i in range(12):
        B_skip_connection = x
        x = tf.keras.layers.Conv3D(C, 3, activation="relu",padding="same",name ='residual_block_{}a'.format(i))(x)
        x = tf.keras.layers.Conv3D(C, 3, activation=None,padding="same",name ='residual_block_{}b'.format(i))(x)
        x = tf.keras.layers.add([x, B_skip_connection])
    x = tf.keras.layers.Conv3D(C, 3, activation="relu",padding="same",name ='conv_2'.format(i))(x)
    x = tf.keras.layers.add([x, skip_connection])

    for i in range(3):
        C = int(x.get_shape()[-1])
        x = tf.keras.layers.Conv3D(C, 3, strides = [1,2,2], activation=tf.keras.layers.LeakyReLU(alpha=0.2),padding="same",
                               name ='c_block_{}a'.format(i))(x)
        x = tf.keras.layers.Conv3D(C_mult[i]*C, 3, activation=tf.keras.layers.LeakyReLU(alpha=0.2),padding="same",
                               name ='compress_block_{}b'.format(i))(x)

    C = int(x.get_shape()[-1])
    encoder_output = tf.keras.layers.Conv3D(int(C/2), 3, activation=None,padding="same",
                               name ='conv_out'.format(i))(x)
    encoder = tf.keras.Model(encoder_input, encoder_output, name="encoder")

    decoder_input = tf.keras.Input(shape=encoder_output.shape[1:], name="encoded_solut")
    C = decoder_input.shape[-1]
    x = tf.keras.layers.Conv3D(C*2, 3, activation=None,padding="same",name='deconv_1')(decoder_input)
    C_div = C_mult[::-1]
    print(C_div)
    for i in range(3):
        C = x.shape[-1]
        C_over_div = int(int(C)/C_div[i])
        print(C_over_div)
        x = tf.keras.layers.Conv3D(C_over_div, 3, activation=tf.keras.layers.LeakyReLU(alpha=0.2),padding="same"
                                   ,name='decompress_block_{}a'.format(i))(x)
        x = tf.keras.layers.Conv3DTranspose(C_over_div, 3, activation=tf.keras.layers.LeakyReLU(alpha=0.2),strides=[1,2,2],padding='same'
                                   ,name='decompress_block_{}b'.format(i))(x)


    skip_connection = x
    C = C_chan
    for i in range(12):
        B_skip_connection = x
        x = tf.keras.layers.Conv3D(C, 3, activation='relu',padding="same",name='deresidual_block_{}a'.format(i))(x)
        x = tf.keras.layers.Conv3D(C, 3, activation=None,padding="same",name ='deresidual_block_{}b'.format(i))(x)
        x = tf.keras.layers.add([x, B_skip_connection])
    x = tf.keras.layers.Conv3D(C, 3, activation=None,padding="same",name ='deconv_2{}'.format(i))(x)
    x = tf.keras.layers.add([x, skip_connection])
    decoder_output = tf.keras.layers.Conv3D(1, 3, activation=None,padding="same",name ='output_solut'.format(i))(x)
    decoder = tf.keras.Model(decoder_input, decoder_output, name="decoder")

    autoencoder_input = tf.keras.Input(shape=input_shape, name="solut")
    encoded_img = encoder(autoencoder_input)
    decoded_img = decoder(encoded_img)
    autoencoder = tf.keras.Model(autoencoder_input, decoded_img, name="autoencoder")
    return autoencoder




# In[37]:


#wrapper methods
log_dir = "./logs"
checkpoint_dir = "./ckpt"

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
    
def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    autoencoder =  build_autoencoder()
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        restored_model =  tf.keras.models.load_model(latest_checkpoint)
        autoencoder.set_weights(restored_model.get_weights())
    print("Creating a new model")

    autoencoder.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=alpha),
        loss=tf.keras.losses.MeanAbsoluteError(reduction='sum_over_batch_size'),
           metrics =[tf.keras.metrics.MeanAbsolutePercentageError()])

    return autoencoder


def run_training(train_dataset,val_dataset,epochs=1):
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Open a strategy scope and create/restore the model
    with strategy.scope():
        model = make_or_restore_model()

    callbacks = [
        # This callback saves a SavedModel every epoch
        # We include the current epoch in the folder name.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"
        ),
        tf.keras.callbacks.CSVLogger(log_dir + "/model_history_log.csv", append=True)
    ]
    model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
        validation_data = val_dataset
    )
    model.save('final_model')




def load_dataset(file_names):
    X = tf.stack(np.array(scipy.io.loadmat(file_names.numpy())["X"],dtype = np.float32))
    y = tf.stack(np.array(scipy.io.loadmat(file_names.numpy())["y"],dtype = np.float32))
    return (X,y)
def load_dataset_wrapper(file_names):
    return tf.py_function(load_dataset, inp=[file_names], Tout=[tf.float32,tf.float32])




def augment(X,y):
    #rotate
    angles = [90, 180, 270, 360]
    angle = random.choice(angles)
    X2 = ndimage.rotate(X, angle, axes=(1, 2), reshape=False)
    y2 = ndimage.rotate(y, angle, axes=(1, 2), reshape=False)
    bools = [True,False]
    boolud = random.choice(bools)
    #flip
    if boolud:
        X2 = X2[:, ::-1, :]
        y2 = y2[:, ::-1, :]
    else:
        X2 = X2
        y2 = y2
    X2 = scaleX(X2)
    y2 = scaleY(y2)
    return (X2,y2)

def augment_wrapper(X,y):
    return tf.py_function(augment, inp=[X,y], Tout=[tf.float32,tf.float32])




def scale_tf(X,y):
    #rotate
    X2 = tf.stack(scaleX(X))
    y2 = tf.stack(scaleY(y))
    return (X2,y2)

def scale_wrapper(X,y):
    return tf.py_function(scale_tf, inp=[X,y], Tout=[tf.float32,tf.float32])




train_paths = get_filepath_list(train_dir+"train"+lossy_e+"/")
val_paths = get_filepath_list(train_dir+"val/")


ds = tf.data.Dataset.from_tensor_slices(train_paths)
ds = (ds
    .shuffle(len(train_paths))
    .map(load_dataset_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    .map(augment_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = tf.data.Dataset.from_tensor_slices(val_paths)
val_ds = (val_ds
    .map(load_dataset_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    .map(scale_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)





run_training(ds,val_ds,epochs)




