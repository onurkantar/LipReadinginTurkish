from PIL import Image, ImageDraw
import face_recognition
import cv2
import numpy as np
import glob
import os
from os import listdir
from os.path import isfile, join
import datetime as dt
import time

data_path = 'C:\\Users\\onur.kantar\\source\\repos\\LipReadinginTurkish\\data\\'

main_path = 'C:\\Users\\onur.kantar\\source\\repos\\LipReadinginTurkish\\ignoreFolder\\'

onlyfiles = [f for f in listdir(main_path) if isfile(join(main_path, f))]

print(onlyfiles)
# Playing video from file:
for path in onlyfiles:
    videoPath , videoName = path.split('_')
    print(videoPath)
    print(videoName)
    innerSavePath = join(data_path,videoPath)
    innerSavePath = innerSavePath + "\\"
    print(innerSavePath)
    videoPath = join(main_path, path)
    print("video path : ")
    print(videoPath)
    cap = cv2.VideoCapture(videoPath)
    try:
        if not os.path.exists(innerSavePath):
            os.makedirs(innerSavePath)
    except OSError:
        print ('Error: Creating directory of data')

    face = 1
    size = 256, 256
    frameNumber = 1
    while(True):

        ret, frame = cap.read()
        if frame is None:
            break
        img = Image.fromarray(frame, 'RGB')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(frame)
        crop_location = (face_locations[face][3],face_locations[face][0],face_locations[face][1],face_locations[face][2])
        crop = img.crop(crop_location)

        crop = crop.resize(size)
        crop = np.array(crop)

        img = np.zeros([256,256,3],dtype=np.uint8)
        pil_image = Image.fromarray(img)
        d = ImageDraw.Draw(pil_image)
        face_landmarks_list = face_recognition.face_landmarks(crop)

        for face_landmarks in face_landmarks_list:
            print("frame {}".format(frameNumber))
            print("The top lip in this face has the following points: {}".format(face_landmarks["top_lip"]))   
            print("The bottom lip in this face has the following points: {}".format(face_landmarks["bottom_lip"]))   
            print("\n")

            d.line(face_landmarks["top_lip"], width=3)
            d.line(face_landmarks["bottom_lip"], width=3)
            name = innerSavePath + videoName.split('.')[0] + "_" + str(frameNumber) + '.jpg'
        pil_image.save(name)
        frameNumber +=1
    cap.release()
