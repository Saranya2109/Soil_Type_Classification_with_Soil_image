#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
import functional_model


# In[5]:


class_names = ['Black soil',
              'Cinder soil',
              'Laterite soil',
              'Peat soil',
              'Yellow soil']


# In[6]:


def predict_soil(path):
    
    img_width, img_height = 150,150   
    im = cv2.imread(path)
    im_resize = cv2.resize(im, (img_width,img_height), interpolation = cv2.INTER_LINEAR)
    
#     image = im_resize.reshape( 1, 150,150,3)
    
    
    plt.imshow(cv2.cvtColor(im_resize, cv2.COLOR_BGR2RGB))
    plt.show()
    
    img_pred = load_img(path,target_size = (img_height, img_width))
    img_pred = img_to_array(img_pred)
    img = np.expand_dims(img_pred, axis = 3)
    image = img.reshape( 1, 150,150,3)
    model = functional_model.get_model()
    result = model.predict(image)
    
    max_confidence = np.argmax(result[0])
    
    print(max_confidence, class_names[max_confidence])

    print('Predicted class', np.argmax(result))


# In[ ]:





# In[ ]:




