# Install Sector4Pooling

Install Sector4Pooling following https://github.com/trucomanx/Sector4Pooling/blob/main/README_install.md 

# Sector4Pooling example code

The next code shows an example use of Sector4Pooling library.

```python
import tensorflow as tf
from Sector4Pooling import Sector4Pooling2D

input_shape=(512, 512,3);

model = tf.keras.Sequential([
    Sector4Pooling2D(factor=0.5,input_shape=input_shape)
])

model.compile(loss='crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


```

# Sector4Pooling example files

Example files can be found at [example.py](example.py).
