import numpy as np
import cv2

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
        blob = cv2.dnn.blobFromImage(frame, 1, (224, 224))

        # here we go!
        net.setInput(blob)

        # forward only - get inference
        output = net.forward()

        # show top 5
        index = np.argsort(output[0])[::-1][:5]

        guess = f'Rank\tID\tLikely\tDescriptin\n'
        for i, id in enumerate(index):
            guess += f'{i+1}\t{id}\t{(output[0][id] * 100):.3}%\t{classes[id]}\n'
        print(guess)
        # frame = cv2.puttext(frame, guess,
        #                     (100, 100),
        #                     cv2.FONT_HERSHEY_SIMPLEX,
        #                     1, (255,255,255), 2)
        
    else:
        break

cap.release()

        