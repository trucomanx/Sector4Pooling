from keras import backend as K
from keras.layers import Layer

import tensorflow as tf
from tensorflow.keras import Sequential
import numpy as np

class custom_layer(Layer):
    def  __init__(self,parameter_dim,**kwargs):
        self.parameter_dim=parameter_dim
        super(custom_layer,self).__init__(**kwargs)
    
    def build(self,input_shape):
        self.W=self.add_weight(name='kernel',
                               shape=(input_shape[1],
                               self.parameter_dim),
                               initializer='uniform',
                               trainable=True)
        self.built = True

    # this self.built is necessary .

    def call(self,x):
        print('x.shape:',x.shape)
        print('self.W.shape:',self.W.shape)
        res=K.dot(x,self.W);
        print('res.shape:',res.shape)
        return res

    def compute_output_shape(self,input_shape):
        return (input_shape[0], self.parameter_dim)
        
################################################################################
d1=5;
d2=7;

input_shape=(d2,);


model = Sequential([
    custom_layer(parameter_dim=3,input_shape=input_shape)
])

model.compile(loss='crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

################################################################################


a=np.linspace(1,d1*d2,d1*d2);
a=a.reshape((d1,d2));

c=tf.constant(a);

print('input:\n',c);

print('output:\n',model.predict(c,verbose=0))


