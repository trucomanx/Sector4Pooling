#!/usr/bin/python

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

import tensorflow as tf
import numpy as np
import math
import sys

def repeat_mat_in_nch_channels(mat,nch):
    tmp=mat.reshape((mat.shape[0],mat.shape[1],1));
    #print('tmp',tmp[:,:,0]);
    for n in range(nch):
        if n==0:
            res=tmp.copy();
        else:
            res=np.concatenate((res,tmp),axis=2);
    return res;


### example https://data-flair.training/blogs/keras-custom-layers/
class Sector4Pooling2D(Layer):
    def  __init__(self,factor=0.5,**kwargs):
        if factor<0.5 or factor >=1.0 :
            sys.error('factor should be <0,1.0>. factor:'+str(factor));
        self.factor=factor;
        super(Sector4Pooling2D,self).__init__(**kwargs)
    
    def build(self,input_shape):
        
        self.Dim1=int(math.ceil(input_shape[1]*self.factor));
        self.Dim2=int(math.ceil(input_shape[2]*self.factor));
        self.Ch  =int(input_shape[3]*4);
        
        Mx=np.eye(self.Dim1);
        My=np.zeros((self.Dim1,input_shape[1]-self.Dim1));
        tmp=np.concatenate((Mx,My),axis=1);
        tmp=repeat_mat_in_nch_channels(tmp,input_shape[3]);
        self.Ia1=tf.constant(tmp, dtype=tf.float32);
        #print('Ia1.shape',self.Ia1.shape);
        
        Mx=np.eye(self.Dim1);
        My=np.zeros((self.Dim1,input_shape[1]-self.Dim1));
        tmp=np.concatenate((My,Mx),axis=1);
        tmp=repeat_mat_in_nch_channels(tmp,input_shape[3]);
        self.Ia2=tf.constant(tmp, dtype=tf.float32);
        #print('Ia2.shape',self.Ia2.shape);
        
        Mx=np.eye(self.Dim2);
        My=np.zeros((input_shape[2]-self.Dim2,self.Dim2));
        tmp=np.concatenate((Mx,My),axis=0);
        tmp=repeat_mat_in_nch_channels(tmp,input_shape[3]);
        self.Ib1=tf.constant(tmp, dtype=tf.float32);
        #print('Ib1.shape',self.Ib1.shape);
        
        Mx=np.eye(self.Dim2);
        My=np.zeros((input_shape[2]-self.Dim2,self.Dim2));
        tmp=np.concatenate((My,Mx),axis=0);
        tmp=repeat_mat_in_nch_channels(tmp,input_shape[3]);
        self.Ib2=tf.constant(tmp, dtype=tf.float32);
        #print('Ib2.shape',self.Ib2.shape);
        
        self.built = True

    # this self.built is necessary .

    def call(self,x):
        #print('');
        
        tmp11=tf.einsum("ebd,abcd->aecd", self.Ia1,x       )
        res11=tf.einsum("abcd,ced->abed", tmp11   ,self.Ib1)
        #print('res11.shape',res11.shape)
        
        tmp12=tf.einsum("ebd,abcd->aecd", self.Ia1,x       )
        res12=tf.einsum("abcd,ced->abed", tmp12   ,self.Ib2)
        #print('res12.shape',res12.shape)
        
        tmp21=tf.einsum("ebd,abcd->aecd", self.Ia2,x       )
        res21=tf.einsum("abcd,ced->abed", tmp21   ,self.Ib1)
        #print('res21.shape',res21.shape)
        
        tmp22=tf.einsum("ebd,abcd->aecd", self.Ia2,x       )
        res22=tf.einsum("abcd,ced->abed", tmp22   ,self.Ib2)
        #print('res22.shape',res22.shape)
        
        out=K.concatenate([res11, res12, res21, res22], axis=3);
        #print('out.shape',out.shape)
        
        return out

    def compute_output_shape(self,input_shape):
        Dim1=int(math.ceil(input_shape[1]*self.factor));
        Dim2=int(math.ceil(input_shape[2]*self.factor));
        Ch  =int(input_shape[3]*4);
        
        output_shape=(input_shape[0],Dim1,Dim2,Ch);
        #print('\noutput_shape\n',output_shape)
        return output_shape;
        
