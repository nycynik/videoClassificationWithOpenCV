import numpy as np
import cv2

# setup models
# read in the models
model_data = open('./models/synset_words.txt').read().strip().split('\n')

classes = [r[r.find(' ') + 1:] for r in model_data]

net = cv2.dnn.readNetFromCaffe(
    './models/bvlc_googlenet.prototxt',
    './models/bvlc_googlenet.caffemodel')

# load video
cap = cv2.VideoCapture('./video/running.mov')

if cap.isOpened() == False:
    raise ('Could not open video')
    
while True:
    ret, frame = cap.read()

    if ret == True:


        