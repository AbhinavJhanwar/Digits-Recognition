# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:51:26 2019

@author: abhinav.jhanwar
"""

import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

import matplotlib.pyplot as plt
image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)

model.evaluate(x_test, y_test)

image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())

#########################################################################################
import cv2
import imutils
import numpy as np

# load image
image = cv2.imread('89.jpg')
#cv2.imshow("number", image)
#cv2.waitKey(0)

# resize
image = imutils.resize(image, height=400)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image.jpg", gray)
#cv2.waitKey(0)

# extrat
thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
#cv2.imshow("Thresh", thresh)
#cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
output = image.copy()
samples = []
# loop over the contours
for c, hier in zip(cnts, hierarchy[0]):
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    if hier[3]==0:
        #cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
        
        # extract contour rectangle boundary
        x, y, w, h = cv2.boundingRect(c)
        if w>20 and h>100:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            samples.append(thresh[y:y+h, x:x+w])

for index, im in enumerate(samples):
    cv2.imshow("Initial Image",im)
    cv2.waitKey(0)
    
    # resize contour to height=28
    im = imutils.resize(im, height=28)
    # add white pixel around the image to convert it into a square
    # 50>> depends on how much area to be covered by number; higher this value lesser space covered by number
    delta_w = 50 - im.shape[1]
    delta_h = 50 - im.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)           
    color = [255]
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    # resize to 28x28 pixel to be fed to neural network
    im = cv2.resize(im, (28,28))
    
    cv2.imshow("Modified Image",im)
    cv2.waitKey(0)
    samples[index] = cv2.threshold(im, 240, 255, cv2.THRESH_BINARY_INV)[1]
    plt.imshow(samples[0],cmap='Greys')
    pred = model.predict(samples[index].reshape(1, 28, 28, 1))
    print(pred.argmax())

# load image
image = cv2.imread('8.jpg')
#cv2.imshow("number", image)
#cv2.waitKey(0)

# resize
image = imutils.resize(image, height=400)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image.jpg", gray)
#cv2.waitKey(0)

# extrat
thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
#cv2.imshow("Thresh", thresh)
#cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
output = image.copy()
samples = []
# loop over the contours
for c, hier in zip(cnts, hierarchy[0]):
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    if hier[3]==0:
        #cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
        
        # extract contour rectangle boundary
        x, y, w, h = cv2.boundingRect(c)
        if w>20 and h>100:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            samples.append(thresh[y:y+h, x:x+w])

for index, im in enumerate(samples):
    cv2.imshow("Initial Image",im)
    cv2.waitKey(0)
    
    im = imutils.resize(im, height=28)
    delta_w = 50 - im.shape[1]
    delta_h = 50 - im.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)           
    color = [255]
    im = cv2.resize(cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=color), (28,28))
    
    cv2.imshow("Modified Image",im)
    cv2.waitKey(0)
    samples[index] = cv2.threshold(im, 240, 255, cv2.THRESH_BINARY_INV)[1]
    plt.imshow(samples[0],cmap='Greys')
    pred = model.predict(samples[index].reshape(1, 28, 28, 1))
    print(pred.argmax())   

###############################################################################################
    
################################################################################################
import cv2
import imutils
from imutils import contours

# load image
image = cv2.imread('sample6.jfif')
#cv2.imshow("number", image)
#cv2.waitKey(0)

# resize
image = imutils.resize(image, height=400)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image.jpg", gray)
#cv2.waitKey(0)

# extract
thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)[1]
#cv2.imshow("Thresh", thresh)
#cv2.waitKey(0)

mask = cv2.erode(thresh, None, iterations=6)
#cv2.imshow("Eroded", mask)
#cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy_levels = np.unique(hierarchy[0,:,3])

#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
output = image.copy()
# loop over the contours
for c, hier in zip(cnts, hierarchy[0]):
    #cv2.drawContours(output, [c], -1, (0, 0, 255), 1)
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    if hier[3]==-1:
        #cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
#cv2.imshow("Contours", output)
#cv2.waitKey(0)
            
        # extract contour rectangle boundary
        x, y, w, h = cv2.boundingRect(c)
        #print(w,h)
        if h>90:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample = mask[y:y+h, x:x+w]
            im = imutils.resize(sample, height=28)
            delta_w = 50 - im.shape[1]
            delta_h = 50 - im.shape[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)           
            color = [0]
            im = cv2.resize(cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color), (28,28))
            pred = model.predict(im.reshape(1, 28, 28, 1))
            cv2.putText(output, str(pred.argmax()), (x+20, y+30), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv2.imshow("Predictions", output)
cv2.waitKey(0)

################################################################################################

################################################################################################

# load image
image = cv2.imread('sample5.jpg')
#cv2.imshow("number", image)
#cv2.waitKey(0)

# resize
image = imutils.resize(image, height=400)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image.jpg", gray)
#cv2.waitKey(0)

# extract
thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)[1]
#cv2.imshow("Thresh", thresh)
#cv2.waitKey(0)

mask = cv2.erode(thresh, None, iterations=1)
mask = cv2.dilate(thresh, None, iterations=1)
#cv2.imshow("Eroded", mask)
#cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy_levels, counts = np.unique(hierarchy[0,:,3],  return_counts =True)

#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
output = image.copy()
# loop over the contours
for c, hier in zip(cnts, hierarchy[0]):
    #cv2.drawContours(output, [c], -1, (0, 0, 255), 1)
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    if hier[3]==-1:
        #cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
#cv2.imshow("Contours", output)
#cv2.waitKey(0)
            
        # extract contour rectangle boundary
        x, y, w, h = cv2.boundingRect(c)
        #print(w,h)
        if h>90:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample = mask[y:y+h, x:x+w]
            im = imutils.resize(sample, height=28)
            delta_w = 50 - im.shape[1]
            delta_h = 50 - im.shape[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)           
            color = [0]
            im = cv2.resize(cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color), (28,28))
            #cv2.imshow("sample", im)
            #cv2.waitKey(0)
            pred = model.predict(im.reshape(1, 28, 28, 1))
            #print(pred.argmax())
            cv2.putText(output, str(pred.argmax()), (x+40, y+40), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv2.imshow("Predictions", output)
cv2.waitKey(0)

################################################################################################

################################################################################################
# load image
image = cv2.imread('sample4.jpg')
#cv2.imshow("number", image)
#cv2.waitKey(0)

# resize
image = imutils.resize(image, height=400)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image.jpg", gray)
#cv2.waitKey(0)

# extract
thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)[1]
#cv2.imshow("Thresh", thresh)
#cv2.waitKey(0)

mask = cv2.erode(thresh, None, iterations=1)
#cv2.imshow("Eroded", mask)
#cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy_levels = np.unique(hierarchy[0,:,3])

#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
output = image.copy()
# loop over the contours
for c, hier in zip(cnts, hierarchy[0]):
    #cv2.drawContours(output, [c], -1, (0, 0, 255), 1)
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    if hier[3]==-1:
        #cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
#cv2.imshow("Contours", output)
#cv2.waitKey(0)
            
        # extract contour rectangle boundary
        x, y, w, h = cv2.boundingRect(c)
        #print(w,h)
        if h>50:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample = mask[y:y+h, x:x+w]
            im = imutils.resize(sample, height=28)
            delta_w = 50 - im.shape[1]
            delta_h = 50 - im.shape[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)           
            color = [0]
            im = cv2.resize(cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color), (28,28))
            #cv2.imshow("sample", im)
            #cv2.waitKey(0)
            pred = model.predict(im.reshape(1, 28, 28, 1))
            cv2.putText(output, str(pred.argmax()), (x+20, y+30), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv2.imshow("Predictions", output)
cv2.waitKey(0)

################################################################################################

################################################################################################
# load image
image = cv2.imread('sample3.jpg')
#cv2.imshow("number", image)
#cv2.waitKey(0)

# resize
image = imutils.resize(image, height=400)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image.jpg", gray)
#cv2.waitKey(0)

# extract
thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)[1]
#cv2.imshow("Thresh", thresh)
#cv2.waitKey(0)

mask = cv2.erode(thresh, None, iterations=6)
#cv2.imshow("Eroded", mask)
#cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy_levels = np.unique(hierarchy[0,:,3])

#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
output = image.copy()
# loop over the contours
for c, hier in zip(cnts, hierarchy[0]):
    #cv2.drawContours(output, [c], -1, (0, 0, 255), 1)
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    if hier[3]==-1:
        #cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
#cv2.imshow("Contours", output)
#cv2.waitKey(0)
            
        # extract contour rectangle boundary
        x, y, w, h = cv2.boundingRect(c)
        #print(w,h)
        if h>50:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample = mask[y:y+h, x:x+w]
            im = imutils.resize(sample, height=28)
            delta_w = 50 - im.shape[1]
            delta_h = 50 - im.shape[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)           
            color = [0]
            im = cv2.resize(cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color), (28,28))
            #cv2.imshow("sample", im)
            #cv2.waitKey(0)
            pred = model.predict(im.reshape(1, 28, 28, 1))
            cv2.putText(output, str(pred.argmax()), (x+20, y+30), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv2.imshow("Predictions", output)
cv2.waitKey(0)

################################################################################################

################################################################################################
# load image
image = cv2.imread('sample2.jpg')
#cv2.imshow("number", image)
#cv2.waitKey(0)

# resize
image = imutils.resize(image, height=400)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image.jpg", gray)
#cv2.waitKey(0)

# extract
thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)[1]
#cv2.imshow("Thresh", thresh)
#cv2.waitKey(0)

mask = cv2.erode(thresh, None, iterations=1)
#cv2.imshow("Eroded", mask)
#cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy_levels = np.unique(hierarchy[0,:,3])

#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
output = image.copy()
# loop over the contours
for c, hier in zip(cnts, hierarchy[0]):
    #cv2.drawContours(output, [c], -1, (0, 0, 255), 1)
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    if hier[3]==-1:
        #cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
#cv2.imshow("Contours", output)
#cv2.waitKey(0)
            
        # extract contour rectangle boundary
        x, y, w, h = cv2.boundingRect(c)
        #print(w,h)
        if h>50:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample = mask[y:y+h, x:x+w]
            im = imutils.resize(sample, height=28)
            delta_w = 50 - im.shape[1]
            delta_h = 50 - im.shape[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)           
            color = [0]
            im = cv2.resize(cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color), (28,28))
            #cv2.imshow("sample", im)
            #cv2.waitKey(0)
            pred = model.predict(im.reshape(1, 28, 28, 1))
            cv2.putText(output, str(pred.argmax()), (x+45, y+45), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv2.imshow("Predictions", output)
cv2.waitKey(0)

######################################################################################################


################################################################################################
# load image
image = cv2.imread('sample1.jpg')
#cv2.imshow("number", image)
#cv2.waitKey(0)

# resize
image = imutils.resize(image, height=400)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image.jpg", gray)
#cv2.waitKey(0)

# extract
thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)[1]
#cv2.imshow("Thresh", thresh)
#cv2.waitKey(0)

mask = cv2.erode(thresh, None, iterations=4)
#cv2.imshow("Eroded", mask)
#cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy_levels = np.unique(hierarchy[0,:,3])

#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
output = image.copy()
# loop over the contours
for c, hier in zip(cnts, hierarchy[0]):
    #cv2.drawContours(output, [c], -1, (0, 0, 255), 1)
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    if hier[3]==-1:
        #cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
#cv2.imshow("Contours", output)
#cv2.waitKey(0)
            
        # extract contour rectangle boundary
        x, y, w, h = cv2.boundingRect(c)
        #print(w,h)
        if h>50:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample = mask[y:y+h, x:x+w]
            im = imutils.resize(sample, height=28)
            #im = cv2.resize(sample, (28,28))
            delta_w = 40 - im.shape[1]
            delta_h = 40 - im.shape[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)           
            color = [0]
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            #im = cv2.dilate(im, None, iterations=1)
            # resize to 28x28 pixel to be fed to neural network
            im = cv2.resize(im, (28,28))
            #cv2.imshow("sample", im)
            #cv2.waitKey(0)
            pred = model.predict(im.reshape(1, 28, 28, 1))
            cv2.putText(output, str(pred.argmax()), (x+20, y+30), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv2.imshow("Predictions", output)
cv2.waitKey(0)

####################################################################################################


################################################################################################
# load image
image = cv2.imread('sample13.jpg')
#cv2.imshow("number", image)
#cv2.waitKey(0)

# resize
image = imutils.resize(image, height=800)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image.jpg", gray)
#cv2.waitKey(0)

# EXTRACTION
edged = cv2.Canny(gray, 100, 255)
#edged = imutils.resize(edged, height=800)
#cv2.imshow("Edged", edged)
#cv2.waitKey(0)

mask = cv2.dilate(edged, None, iterations=2)
mask = cv2.erode(mask, None, iterations=2)
# resize
#mask = imutils.resize(mask, height=800)
#cv2.imshow("Eroded", mask)
#cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy_levels = np.unique(hierarchy[0,:,3])

#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
output = image.copy()
# loop over the contours
for c, hier in zip(cnts, hierarchy[0]):
    #cv2.drawContours(output, [c], -1, (0, 0, 255), 1)
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    if hier[3]==-1:
        #x, y, w, h = cv2.boundingRect(c)
        #cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
        #cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
#cv2.imshow("Contours", output)
#cv2.waitKey(0)
            
        # extract contour rectangle boundary
        x, y, w, h = cv2.boundingRect(c)
        #print(w,h)
        if h>35:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample = mask[y:y+h, x:x+w]
            im = imutils.resize(sample, height=28)
            #im = cv2.erode(im, None, iterations=1)
            #im = cv2.resize(sample, (28,28))
            delta_w = 48 - im.shape[1]
            delta_h = 48 - im.shape[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)           
            color = [0]
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            #im = cv2.dilate(im, None, iterations=1)
            # resize to 28x28 pixel to be fed to neural network
            im = cv2.resize(im, (28,28))
            #cv2.imshow("sample", im)
            #cv2.waitKey(0)
            pred = model.predict(im.reshape(1, 28, 28, 1))
            #print(pred.argmax())
            cv2.putText(output, str(pred.argmax()), (x+20, y), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv2.imshow("Predictions", output)
cv2.waitKey(0)

################################################################################################


################################################################################################
# load image
image = cv2.imread('sample11.jpg')
#cv2.imshow("number", image)
#cv2.waitKey(0)

# resize
image = imutils.resize(image, height=800)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image.jpg", gray)
#cv2.waitKey(0)

# EXTRACTION
edged = cv2.Canny(gray, 100, 120)
#edged = imutils.resize(edged, height=800)
#cv2.imshow("Edged", edged)
#cv2.waitKey(0)

mask = cv2.dilate(edged, None, iterations=2)
mask = cv2.erode(mask, None, iterations=2)
# resize
#mask = imutils.resize(mask, height=800)
#cv2.imshow("Eroded", mask)
#cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy_levels = np.unique(hierarchy[0,:,3])

#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
output = image.copy()
# loop over the contours
for c, hier in zip(cnts, hierarchy[0]):
    #cv2.drawContours(output, [c], -1, (0, 0, 255), 1)
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    if hier[3]==-1:
        #x, y, w, h = cv2.boundingRect(c)
        #cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
        #cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
#cv2.imshow("Contours", output)
#cv2.waitKey(0)
            
        # extract contour rectangle boundary
        x, y, w, h = cv2.boundingRect(c)
        #print(w,h)
        if h>35:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample = mask[y:y+h, x:x+w]
            im = imutils.resize(sample, height=28)
            #im = cv2.erode(im, None, iterations=1)
            #im = cv2.resize(sample, (28,28))
            delta_w = 48 - im.shape[1]
            delta_h = 48 - im.shape[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)           
            color = [0]
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            #im = cv2.dilate(im, None, iterations=1)
            # resize to 28x28 pixel to be fed to neural network
            im = cv2.resize(im, (28,28))
            #cv2.imshow("sample", im)
            #cv2.waitKey(0)
            pred = model.predict(im.reshape(1, 28, 28, 1))
            #print(pred.argmax())
            cv2.putText(output, str(pred.argmax()), (x+20, y), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv2.imshow("Predictions", output)
cv2.waitKey(0)

###########################################################################################



################################################################################################

# load image
image = cv2.imread('sample8.jpg')
#cv2.imshow("number", image)
#cv2.waitKey(0)

# resize
image = imutils.resize(image, height=800)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image.jpg", gray)
#cv2.waitKey(0)

# EXTRACTION
edged = cv2.Canny(gray, 100, 120)
#cv2.imshow("Edged", edged)
#cv2.waitKey(0)

mask = cv2.dilate(edged, None, iterations=2)
mask = cv2.erode(mask, None, iterations=2)
#cv2.imshow("Eroded", mask)
#cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy_levels = np.unique(hierarchy[0,:,3])

#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
output = image.copy()
# loop over the contours
for c, hier in zip(cnts, hierarchy[0]):
    #cv2.drawContours(output, [c], -1, (0, 0, 255), 1)
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    if hier[3]==-1:
        #x, y, w, h = cv2.boundingRect(c)
        #cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
        #cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
#cv2.imshow("Contours", output)
#cv2.waitKey(0)
            
        # extract contour rectangle boundary
        x, y, w, h = cv2.boundingRect(c)
        #print(w,h)
        if h>30:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample = mask[y:y+h, x:x+w]
            im = imutils.resize(sample, height=28)
            #im = cv2.erode(im, None, iterations=1)
            #im = cv2.resize(sample, (28,28))
            delta_w = 48 - im.shape[1]
            delta_h = 48 - im.shape[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)           
            color = [0]
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            im = cv2.dilate(im, None, iterations=1)
            # resize to 28x28 pixel to be fed to neural network
            im = cv2.resize(im, (28,28))
            #cv2.imshow("sample", im)
            #cv2.waitKey(0)
            pred = model.predict(im.reshape(1, 28, 28, 1))
            #print(pred.argmax())
            cv2.putText(output, str(pred.argmax()), (x+20, y), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv2.imshow("Predictions", output)
cv2.waitKey(0)

###########################################################################################


################################################################################################

# load image
image = cv2.imread('sample9.jpg')
#cv2.imshow("number", image)
#cv2.waitKey(0)

# resize
image = imutils.resize(image, height=800)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image.jpg", gray)
#cv2.waitKey(0)

# EXTRACTION
edged = cv2.Canny(gray, 100, 120)
#cv2.imshow("Edged", edged)
#cv2.waitKey(0)

mask = cv2.dilate(edged, None, iterations=2)
mask = cv2.erode(mask, None, iterations=2)
#cv2.imshow("Eroded", mask)
#cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy_levels = np.unique(hierarchy[0,:,3])

#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
output = image.copy()
# loop over the contours
for c, hier in zip(cnts, hierarchy[0]):
    #cv2.drawContours(output, [c], -1, (0, 0, 255), 1)
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    if hier[3]==-1:
        #x, y, w, h = cv2.boundingRect(c)
        #cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
        #cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
#cv2.imshow("Contours", output)
#cv2.waitKey(0)
            
        # extract contour rectangle boundary
        x, y, w, h = cv2.boundingRect(c)
        #print(w,h)
        if h>30:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample = mask[y:y+h, x:x+w]
            im = imutils.resize(sample, height=28)
            #im = cv2.erode(im, None, iterations=1)
            #im = cv2.resize(sample, (28,28))
            delta_w = 48 - im.shape[1]
            delta_h = 48 - im.shape[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)           
            color = [0]
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            im = cv2.dilate(im, None, iterations=1)
            # resize to 28x28 pixel to be fed to neural network
            im = cv2.resize(im, (28,28))
            #cv2.imshow("sample", im)
            #cv2.waitKey(0)
            pred = model.predict(im.reshape(1, 28, 28, 1))
            #print(pred.argmax())
            cv2.putText(output, str(pred.argmax()), (x+20, y), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv2.imshow("Predictions", output)
cv2.waitKey(0)

###########################################################################################


################################################################################################

# load image
image = cv2.imread('sample12.jpg')
#cv2.imshow("number", image)
#cv2.waitKey(0)

# resize
image = imutils.resize(image, height=800)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image.jpg", gray)
#cv2.waitKey(0)

# EXTRACTION
edged = cv2.Canny(gray, 210, 230)
#cv2.imshow("Edged", edged)
#cv2.waitKey(0)

mask = cv2.dilate(edged, None, iterations=1)
mask = cv2.erode(mask, None, iterations=1)
#cv2.imshow("Eroded", mask)
#cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy_levels = np.unique(hierarchy[0,:,3])

#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
output = image.copy()
# loop over the contours
for c, hier in zip(cnts, hierarchy[0]):
    #cv2.drawContours(output, [c], -1, (0, 0, 255), 1)
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    if hier[3]==-1:
        #x, y, w, h = cv2.boundingRect(c)
        #cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
        #cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
#cv2.imshow("Contours", output)
#cv2.waitKey(0)
            
        # extract contour rectangle boundary
        x, y, w, h = cv2.boundingRect(c)
        #print(w,h)
        if h>15:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample = mask[y:y+h, x:x+w]
            im = imutils.resize(sample, height=28)
            #im = cv2.erode(im, None, iterations=1)
            #im = cv2.resize(sample, (28,28))
            delta_w = 48 - im.shape[1]
            delta_h = 48 - im.shape[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)           
            color = [0]
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            im = cv2.dilate(im, None, iterations=1)
            # resize to 28x28 pixel to be fed to neural network
            im = cv2.resize(im, (28,28))
            #cv2.imshow("sample", im)
            #cv2.waitKey(0)
            pred = model.predict(im.reshape(1, 28, 28, 1))
            #print(pred.argmax())
            cv2.putText(output, str(pred.argmax()), (x+20, y), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv2.imshow("Predictions", output)
cv2.waitKey(0)

###########################################################################################


################################################################################################

# load image
image = cv2.imread('sample13.jpg')
#cv2.imshow("number", image)
#cv2.waitKey(0)

# resize
image = imutils.resize(image, height=800)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image.jpg", gray)
#cv2.waitKey(0)

# EXTRACTION
edged = cv2.Canny(gray, 100, 170)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

mask = cv2.dilate(edged, None, iterations=1)
mask = cv2.erode(mask, None, iterations=1)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)

cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy_levels = np.unique(hierarchy[0,:,3])

#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
output = image.copy()
# loop over the contours
for c, hier in zip(cnts, hierarchy[0]):
    #cv2.drawContours(output, [c], -1, (0, 0, 255), 1)
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    if hier[3]==-1:
        #x, y, w, h = cv2.boundingRect(c)
        #cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
        #cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
#cv2.imshow("Contours", output)
#cv2.waitKey(0)
            
        # extract contour rectangle boundary
        x, y, w, h = cv2.boundingRect(c)
        #print(w,h)
        if h>10:# and w<100:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample = mask[y:y+h, x:x+w]
            if h>w:
                im = imutils.resize(sample, height=28)
            else:
                im = imutils.resize(sample, width=28)
            #im = cv2.erode(im, None, iterations=1)
            #im = cv2.resize(sample, (28,28))
            delta_w = 48 - im.shape[1]
            delta_h = 48 - im.shape[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)           
            color = [0]
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            #im = cv2.dilate(im, None, iterations=1)
            # resize to 28x28 pixel to be fed to neural network
            im = cv2.resize(im, (28,28))
            #cv2.imshow("sample", im)
            #cv2.waitKey(0)
            pred = model.predict(im.reshape(1, 28, 28, 1))
            #print(pred.argmax())
            cv2.putText(output, str(pred.argmax()), (x+20, y), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv2.imshow("Predictions", output)
cv2.waitKey(0)

