# -*- coding: utf-8 -*-

import random
import numpy as np
import tensorflow as tf
from keras.layers import Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model
from keras import backend as K

def siamese_model_attentiveFp_sider ():

    left_input = Input(shape=(1024,))
    right_input = Input(shape=(512,))

    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1024,input_shape=(1024,),  activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),

    ])

    model2 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512,input_shape=(512,),  activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
    ])

    encoded_l = model1(left_input)
    encoded_r = model2(right_input)

    L1 = tf.keras.layers.Lambda (lambda x: K.abs(x[0]-x[1]))([encoded_l, encoded_r])

    L1_D = tf.keras.layers.Dropout(0.2)(L1)
    L2 = tf.keras.layers.Dense(64, activation='relu')(L1_D)
    L2_D = tf.keras.layers.Dropout(0.2)(L2)
    L3 = tf.keras.layers.Dense(32, activation='relu')(L2_D)
    L3_D = tf.keras.layers.Dropout(0.2)(L3)
    L4 = tf.keras.layers.Dense(16, activation='relu')(L3_D)
    L4_D = tf.keras.layers.Dropout(0.2)(L4)
    L5 = tf.keras.layers.Dense(8, activation='relu')(L4_D)

    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(L5)

    siamese_net = tf.keras.Model([left_input, right_input], prediction)

    optimizer= Adam(learning_rate=0.001)
    siamese_net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy", "mae", "mse",tf.keras.metrics.AUC()])

    optimizer = Adam(learning_rate = 0.001)
    model1.compile(loss="kl_divergence",optimizer=optimizer)
    model2.compile(loss="kl_divergence",optimizer=optimizer)

    return siamese_net

def siamese_model_Canonical_sider (lr_1 = 0.001 , lr_2 = 0.001):

    left_input = Input(shape=(512,))
    right_input = Input(shape=(512,))

    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512,input_shape=(512,),  activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
    ])

    model2 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512,input_shape=(512,),  activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
    ])

    encoded_l = model1(left_input)
    encoded_r = model2(right_input)

    L1 = tf.keras.layers.Lambda (lambda x: K.abs(x[0]-x[1]))([encoded_l, encoded_r])

    L1_D = tf.keras.layers.Dropout(0.2)(L1)
    L2 = tf.keras.layers.Dense(64, activation='relu')(L1_D)
    L2_D = tf.keras.layers.Dropout(0.2)(L2)
    L3 = tf.keras.layers.Dense(32, activation='relu')(L2_D)
    L3_D = tf.keras.layers.Dropout(0.2)(L3)
    L4 = tf.keras.layers.Dense(16, activation='relu')(L3_D)
    L4_D = tf.keras.layers.Dropout(0.2)(L4)
    L5 = tf.keras.layers.Dense(8, activation='relu')(L4_D)

    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(L5)

    siamese_net = tf.keras.Model([left_input, right_input], prediction)

    optimizer= Adam(learning_rate = lr_1)
    siamese_net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy", "mae", "mse",tf.keras.metrics.AUC()])

    optimizer = Adam(learning_rate = lr_2)
    model1.compile(loss="kl_divergence",optimizer=optimizer)
    model2.compile(loss="kl_divergence",optimizer=optimizer)

    return siamese_net