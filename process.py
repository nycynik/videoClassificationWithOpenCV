import numpy as np
import cv2
import VideoClassifier
import argparse
import os

"""
Video Classifer runner
Video Classification

Michael Lynch <https://github.com/nycynik>
"""


def main():

    parser = argparse.ArgumentParser(
        description='Process a video and name objects found in it.')
    parser.add_argument('input',
                        help='the input video file path')
    parser.add_argument('-o', '--output', default='out',
                        help='the output video file path')

    args = parser.parse_args()

    # == verify args ==
    # source
    source = None
    if args.input is not None and os.path.isfile(args.input):
        source = args.input
    else:
        raise Exception(
            'input file does not exist, or is not a file. Please update your input file.')

    # destination
    destination = 'output.mp4'
    if args.output is not None and os.path.isfile(args.output):
        raise Exception(
            'output file exists, and not set to overwrite. Please update your output file.')

    if args.output is not None:
        destination = args.output

    # Let's go!
    vc = VideoClassifier.VideoClassifier(source, destination)
    vc.process()


if __name__ == "__main__":
    main()
