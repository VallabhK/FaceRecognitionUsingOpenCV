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

# Load the stored face detector model
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# Load the corresponding face embeddings
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

