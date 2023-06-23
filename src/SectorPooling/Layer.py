#!/usr/bin/python

import sys
import os


import numpy as np
import tensorflow as tf


################################################################################
from Pooling2D import SectorNPooling2D
from tensorflow.keras import backend as K
import time

def layersectortree2d_recursive_func(entrada,blocks,factor,min_size,name):
    work=[None,None,None,None];
    out=[];
    
    for n in range(4):
        tmp = SectorNPooling2D(factor=factor,sector=n)(entrada);
        
        model_block=tf.keras.models.clone_model(blocks[0]);
        
        model_block._name=name+str(n);
        
        tmp = model_block(tmp);
        
        new_blocks=blocks;
        if len(blocks)>1:
            new_blocks=blocks[1:];
        
        if tmp.shape[1]*factor>=min_size and tmp.shape[2]*factor>=min_size:
            work[n]=layersectortree2d_recursive_func(tmp,new_blocks,factor,min_size,name+str(n));
        else:
            work[n]=[tmp];
        
        out=out+work[n];
    
    return out;


def LayerSectorTree2D(input_shape,blocks,factor=0.618,min_size=9,name=None,to_file=None):
    if not isinstance(name, str):
        name=str(time.time_ns())+'_';
    
    entrada = tf.keras.layers.Input(shape=input_shape);
    
    out=layersectortree2d_recursive_func(entrada,blocks,factor,min_size,name);
    
    salida=K.concatenate(out, axis=3);
    
    model = tf.keras.Model(inputs=entrada, outputs=salida)
    
    if not isinstance(to_file, str):
        tf.keras.utils.plot_model(model, to_file=to_file,dpi=200,show_shapes=True)
    
    return model


################################################################################

import tensorflow as tf


from tensorflow.keras.layers import LeakyReLU

block1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D( 16, kernel_size=9, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  8, kernel_size=9, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  4, kernel_size=9, padding="same", activation=LeakyReLU()),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D( 16, kernel_size=7, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  8, kernel_size=7, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  4, kernel_size=7, padding="same", activation=LeakyReLU()),
])

block2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D( 16, kernel_size=5, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  8, kernel_size=5, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  4, kernel_size=5, padding="same", activation=LeakyReLU()),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D( 16, kernel_size=3, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  8, kernel_size=3, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  4, kernel_size=3, padding="same", activation=LeakyReLU()),
])

factor=0.618;
min_size=5;
input_shape=(128,128,3);


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D( 3, kernel_size=9, padding="same", activation=LeakyReLU(), input_shape=input_shape),
    LayerSectorTree2D(  input_shape=(128,128,3),
                        blocks=[block1,block2],
                        factor=factor,
                        min_size=min_size,
                        name='LST_',
                        to_file='layer_tree.png'),
    tf.keras.layers.Conv2D( 64, kernel_size=1, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32),
])

model.summary()



