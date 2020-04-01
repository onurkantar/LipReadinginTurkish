# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import json

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"]) #iBUG 300-W dataset.

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)
face_count = 1
data=[]

for (i, rect) in enumerate(rects):
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	face = {"face": face_count}
	face_count +=1

	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		landmark = []
		coordinates = []
		for (x, y) in shape[i:j]:
			landmark.append({x,y})
		face[name] = landmark
	data.append(face)
print(data)