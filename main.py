from PIL import Image, ImageDraw
import face_recognition
import cv2
import numpy as np
import glob
import os
 
# Playing video from file:
cap = cv2.VideoCapture('./ignoreFolder/testVideo.mp4')
try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

frameNumber = 1
while(True):
    # Capture frame-by-frame
    
    ret, frame = cap.read()
    if frame is None:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_locations = face_recognition.face_locations(frame)
    
    img = np.zeros([720,1080,3],dtype=np.uint8)
    pil_image = Image.fromarray(img)
    d = ImageDraw.Draw(pil_image)
    face_landmarks_list = face_recognition.face_landmarks(frame)
    face_landmarks = face_landmarks_list[0]
    print("frame {}".format(frameNumber))
    print("The top lip in this face has the following points: {}".format(face_landmarks["top_lip"]))   
    print("The bottom lip in this face has the following points: {}".format(face_landmarks["bottom_lip"]))   
    print("\n")
        
    d.line(face_landmarks["top_lip"], width=5)
    d.line(face_landmarks["bottom_lip"], width=5)
    name = './data/frame' + str(frameNumber) + '.jpg'
    pil_image.save(name)
    frameNumber +=1
cap.release()
