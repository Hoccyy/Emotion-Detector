import os
import cv2
import imghdr
import tensorflow as tf
import modelFuncs
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

def takePic():
    cap = cv2.VideoCapture(0)
    i = 1
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Operations on the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('Webcam Feed', gray_frame)

        # Save required number of images
        if i < 3 and i != 1:
            img_name = 'imageForUse' + str(i) + '.png'
            cv2.imwrite(img_name, gray_frame)
        
        i = i + 1

        if i > 3 or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return 'imageForUse' + str(i) + '.png'

def main():
    takeNewPic = input("Take a new photo (Y) or enter filename (N) : ")
    if takeNewPic[0] in ['y', 'Y']:
        takePic()
        img = cv2.imread('imageForUse2.png')
        
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
            return
    


    # Load image
    #img = cv2.imread('web3-happy-people-outside-smile-sun-nature-eduardo-dutra-620857-unsplash.jpg')
    img = cv2.imread(input('Enter filename of photo : '))
    resize = tf.image.resize(img, (256,256))

    print ('1313')
    print(img)


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

main()