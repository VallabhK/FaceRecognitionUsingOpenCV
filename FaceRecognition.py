#Problem Statement: Detect faces in the image/video and recognize them
#in real time

#Import all the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2 #OpenCV package
import os

#Let's deal with arguments passed from the command line. These arguments
#will give you the input data and other attributes

argumentparse = argparse.ArgumentParser()
argumentparse.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
argumentparse.add_argument("-e", "--embeddings", required=True,
	help="path to output serialized db of facial embeddings")
argumentparse.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
argumentparse.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
argumentparse.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(argumentparse.parse_args())

