# Install Sector4Pooling

Install Sector4Pooling following https://github.com/trucomanx/Sector4Pooling/blob/main/README_install.md 

# Sector4Pooling example code

The next code shows an example use of Sector4Pooling library.

```python


####################
# Variables

import numpy as np

dim1=32;
dim2=32;
nch=2;

a=np.linspace(1,dim1*dim2*nch,dim1*dim2*nch);

input_data  = a.reshape((1,dim1,dim2,nch));

####################
# Creating the model

import tensorflow as tf
from Sector4Pooling import Sector4Pooling2D

input_shape=(dim1, dim2,nch);

model = tf.keras.Sequential([
    Sector4Pooling2D(factor=0.6,input_shape=input_shape)
])

model.compile(loss='crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

####################
# Applying the model

output_data = model.predict(input_data);

```

# Sector4Pooling example files

Example files can be found at [example.py](example.py).
