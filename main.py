import os
import cv2
import imghdr
import tensorflow as tf
import modelFuncs
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

# Load image
img = cv2.imread('web3-happy-people-outside-smile-sun-nature-eduardo-dutra-620857-unsplash.jpg')
resize = tf.image.resize(img, (256,256))


new_model = load_model('models/imageclassifier.h5')
emotion_prediction = new_model.predict(np.expand_dims(resize/255, 0))

if emotion_prediction > 0.5: 
    print(f'Predicted classification is Sad')
    print(emotion_prediction)
    plt.imshow(img)
    plt.suptitle('This person is Sad 😢' )
    plt.show()
else:
    print(f'Predicted classification is Happy')
    print(emotion_prediction)
    plt.imshow(img)
    plt.suptitle('This person is Happy 😁' )
    plt.show()