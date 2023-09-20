#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Activation,Dropout,MaxPool2D,Flatten,Conv2D
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adagrad


# In[2]:


def get_model():
    
    img_width, img_height = 150,150
    inputs = keras.Input(shape=(150,150,3))
    x = layers.Conv2D(64,3,3, padding = 'same', activation = 'relu')(inputs)
    x = layers.MaxPool2D(pool_size = (2,2))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(32,3,3, padding = 'same', activation = 'relu')(x)
    x = layers.MaxPool2D(pool_size = (2,2))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation = 'relu')(x)
    x = Flatten()(x)
    x = layers.Dense(32, activation = 'relu')(x)
    outputs = layers.Dense(5, activation = 'softmax')(x)
    
    func_model = Model(inputs = inputs, outputs = outputs)
    
    func_model.summary()
    
    func_model.compile(optimizer='adam', loss ='categorical_crossentropy', metrics = ['accuracy'])
    
    
    return func_model


# In[ ]:




