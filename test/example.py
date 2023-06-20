#!/usr/bin/python
import sys

sys.path.append('../src')

from Sector4Pooling import Sector4Pooling2D

################################################################################
from tensorflow.keras import Sequential

d1=32;
d2=32;
ch=2;

input_shape=(d1, d2,ch);

model = Sequential([
    Sector4Pooling2D(factor=0.6,input_shape=input_shape)
])

model.compile(loss='crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

################################################################################

import numpy as np
a=np.linspace(1,d1*d2*ch,d1*d2*ch);

input_data  = a.reshape((1,d1,d2,ch));
output_data = model.predict(input_data);

#print('input_data:\n',input_data);
print('input_data.shape:\n',input_data.shape);
#print('output_data:\n',output_data);
print('output_data.shape:\n',output_data.shape);


