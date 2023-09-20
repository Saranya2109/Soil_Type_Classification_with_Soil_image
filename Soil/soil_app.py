#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import jsonify,Flask, Request
from flask_restful import Api, Resource
from keras.models import load_model
import cv2
import tensorflow
from tensorflow.keras.utils import load_img, img_to_array


# In[5]:


pip install flask_cors


# In[ ]:


import numpy as np
from PIL import Image
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)
CORS(app)

model = load_model('soil-model.h5')

@app.route('/app')
def index():
    return "App is working"



@app.route('/imgrec', methods=['GET', 'POST'])
def image_recognize():
    
       
    class_names = ['Black soil',
              'Cinder soil',
              'Laterite soil',
              'Peat soil',
              'Yellow soil']   
    
    img_file = request.files['img']
    # Read the image via file.stream
    img = Image.open(img_file.stream)
    im_resize = cv2.resize(img, (img_width,img_height), interpolation = cv2.INTER_LINEAR)
    img = np.array(img)
    img_pred = load_img(path,target_size = (img_height, img_width))
    img_pred = img_to_array(img_pred)
    img = np.expand_dims(img_pred, axis = 3)
    image = img.reshape( 1, 150,150,3)
    resp = model.predict(image)
    
    return (class_names[np.argmax(resp[0])])

if __name__ == "__main__":
    app.run(debug=False)


# In[ ]:





# In[ ]:




