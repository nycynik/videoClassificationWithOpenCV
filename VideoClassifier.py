import numpy as np
import cv2

"""
Video Classifer class
Video Classification

Michael Lynch <https://github.com/nycynik>
"""


class VideoClassifier(object):

    def __init__(self, source, destination, **kwargs):
        """Initializes the object

        """
        self.source = source
        self.destination = destination
        self.debug = kwargs.get('debug', False)

    def process(self):

        # read in the models
        # TODO: refactor path to models to be a param kwargs.
        model_data = open(
            './models/synset_words.txt').read().strip().split('\n')

        classes = [r[r.find(' ') + 1:] for r in model_data]

        net = cv2.dnn.readNetFromCaffe(
            './models/bvlc_googlenet.prototxt',
            './models/bvlc_googlenet.caffemodel')

        # load video
        cap = cv2.VideoCapture(self.source)

        if cap.isOpened() == False:
            raise ('Could not open video')

        # prime the pump - we need the first frame to get sizes.
        ret, frame = cap.read()
        frame_size = frame.shape[::2]

        # create output video
        out = cv2.VideoWriter(self.destination,
                              cv2.VideoWriter_fourcc(*'XVID'),
                              10, frame_size)

        while True:

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

                frame = cv2.putText(frame, guess,
                                    (100, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 255, 255), 2)

                # save
                out.write(frame)

                ret, frame = cap.read()

            else:
                break

        cap.release()
        out.release()
