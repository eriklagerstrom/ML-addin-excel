import xlwings as xw
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras


@xw.func
def first_function():
    """very complex function"""
    return "fuck!"
    

@xw.func
@xw.arg("train_x", np.array)
@xw.arg("train_y", np.array)
@xw.arg("pred_x", np.array)
def lin_reg(train_x, train_y, pred_x):

    train_x = train_x.astype(np.float)
    train_y = train_y.astype(np.float)
    pred_x = pred_x.astype(np.float)

    normalizer = preprocessing.Normalization(input_shape = [train_x.shape[1],])
    normalizer.adapt(train_x)
   
    model = tf.keras.Sequential([
        normalizer, layers.Dense(units=1)
    ])
    model.compile(
        optimizer = tf.optimizers.Adam(learning_rate = 0.05),
        loss = 'mean_absolute_error')
    model.fit(
        train_x, train_y, epochs = 20, verbose = 0
    )
    
    #What about normalizing pred_x?
    ret = model.predict(pred_x)
    
    return ret

@xw.func
@xw.arg("train_x", np.array)
@xw.arg("train_y", np.array)
@xw.arg("pred_x", np.array)
def dnn(train_x, train_y, pred_x, layer_1_nodes, layer_2_nodes):

    train_x = train_x.astype(np.float)
    train_y = train_y.astype(np.float)
    pred_x = pred_x.astype(np.float)

    normalizer = preprocessing.Normalization(input_shape = [train_x.shape[1],])
    normalizer.adapt(train_x)
   
    model = tf.keras.Sequential([
        normalizer, layers.Dense(layer_1_nodes, activation = 'relu'),
        layers.Dense(layer_2_nodes, activation = 'relu'), layers.Dense(1)
    ])
    model.compile(
        optimizer = tf.optimizers.Adam(learning_rate = 0.01),
        loss = 'mean_absolute_error')
    model.fit(
        train_x, train_y, epochs = 30, verbose = 0
    )
    
    #What about normalizing pred_x?
    ret = model.predict(pred_x)
    
    return ret
    
    
    


