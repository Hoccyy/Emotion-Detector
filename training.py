import os
import cv2
import imghdr
import modelFuncs
import tensorflow as tf

# Fetch device GPUs for performance
GPUs = tf.config.experimental.list_physical_devices('GPU')
print("GPUs available:{}".format(GPUs))

# Prevent overuse
for gpu in GPUs:
    tf.config.experimental.set_memory_growth(gpu, True)

# Directory for training data classes with training data
data_dir = 'data'

# List of common image extensions
image_exts = ['jpeg', 'jpg', 'png', 'bmp', 'gif', 'svg']
filtration_db = ['.DS_Store']

# Validity check of photos in terms of file types and naming schemes
for image_class in os.listdir(data_dir):
    if image_class not in filtration_db:
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                file_ext = imghdr.what(image_path)

                if file_ext and file_ext.strip() not in image_exts:
                    print('Check Image format\n{}'.format(image_path))
                    os.remove(image_path)
                if not file_ext:
                    c = ''

                    for char in image_path[::-1]:
                        if char == '.':
                            break
                        c += (char) if '.' not in c else None

                    if (c[::-1] not in image_exts) and (c not in image_exts) and (c.lower() not in image_exts) and (c[::-1] not in image_exts):
                        raise
            except Exception as e:
                print("{}, Issue with image\n{}".format(e, image_path))

# Load dataset of images
tf.data.Dataset
import numpy as np
from matplotlib import pyplot as plt

data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

data = data.map(lambda x,y: (x/255, y))

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

print(model.summary())

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])


# Show model Accuracy - TODO
def modelAccuracy():
    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()


# Show model loss - TODO
def modelLoss():
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

modelAccuracy()
modelLoss()

from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(pre.result(), re.result(), acc.result())


# Testing on files
img = cv2.imread('THAEIAATMO.JPG')
# plt.imshow(img)
# plt.show()

resize = tf.image.resize(img, (256,256))
# plt.imshow(resize.numpy().astype(int))
# plt.show()

# Model serialization
from tensorflow.keras.models import load_model
model.save(os.path.join('models','imageclassifier3.h5'))


predic_value = model.predict(np.expand_dims(resize/255, 0))

if predic_value > 0.5: 
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')