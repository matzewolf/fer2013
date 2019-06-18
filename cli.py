#!/usr/bin/env python3

from __future__ import division
import tensorflow as tf
from keras.models import model_from_json
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches


# disable Tensorflow warnings
tf.logging.set_verbosity(tf.logging.ERROR)

# loading the model
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("fer.h5")
print("Loaded model from disk.")

# setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x = None
y = None

# defining labels
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# loading image and convert to gray
full_size_image = cv2.imread(sys.argv[1])  # second cli argument is the image!
print("Image loaded: " + sys.argv[1])
gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)

# detect face(s)
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face.detectMultiScale(gray, 1.3, 10)
print(str(len(faces)) + " face(s) detected.")

# convert cv2's BGR to matplotlib's RGB system
full_size_image = full_size_image[:, :, ::-1]

# create figure and axes
fig, ax = plt.subplots(1)

# Display image
ax.imshow(full_size_image)

# operate on each detected face
for (x, y, w, h) in faces:
    roi_gray = gray[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)

    # draw rectangle around each face
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # classify emotion
    yhat = loaded_model.predict(cropped_img)
    emotion = labels[int(np.argmax(yhat))]

    # print result in image plot and command line
    plt.text(x, y, emotion, fontsize=16, color='red')
    print("Emotion: " + emotion)

plt.show()
