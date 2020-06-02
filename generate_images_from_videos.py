from PIL import Image, ImageDraw
import cv2
import numpy as np
import glob
import os
import model
from enviroment import detector, predictor, main_path, data_path
import dlib
from os import listdir
from os.path import isfile, join
import datetime as dt
import time
from random import shuffle

def generate_images():
    onlyfiles = [f for f in listdir(main_path) if isfile(join(main_path, f))]
    print(onlyfiles)
    # Playing video from file:
    for path in onlyfiles:
        videoPath , videoName = path.split('_')
        innerSavePath = join(data_path,videoPath)
        innerSavePath = innerSavePath + "\\"
        videoPath = join(main_path, path)
        cap = cv2.VideoCapture(videoPath)
        try:
            if not os.path.exists(innerSavePath):
                os.makedirs(innerSavePath)
        except OSError:
            print ('Error: Creating directory of data')
    
        size = 256, 256
        frameNumber = 1
        photos = []
        while(True):
        
            ret, frame = cap.read()
            if frame is None:
                break
            
            if frameNumber % 2 != 0:
                frameNumber += 1
                continue
            else:
                print(frameNumber)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_image = clahe.apply(gray)
    
            detections = detector(clahe_image, 1) #Detect the faces in the image
    
            for k,d in enumerate(detections): #For each detected face
                shape = predictor(clahe_image, d) #Get coordinates
                
            x = []
            y = []
            for i in range(50,68):
                x.append(shape.part(i).x)
                y.append(shape.part(i).y)
    
            img = Image.fromarray(frame, 'RGB')
            crop = img.crop((min(x)-20,min(y)-20,max(x)+20,max(y)+20))
            #crop = crop.resize(size)

            gray = cv2.cvtColor(np.array(crop), cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_image2 = clahe.apply(gray)
            
            cv2.imwrite('b.png',clahe_image2)
            cv2.imwrite('a.png',gray)
            
            
            photos.append(clahe_image2)
            frameNumber +=1

            if frameNumber > 80:
                break
        
        while frameNumber <= 80 :
            
            if frameNumber % 2 != 0:
                frameNumber += 1

            image = np.zeros([256,256],dtype=np.uint8)
            photos.append(image)
            frameNumber +=2

        np.save(innerSavePath + videoName.split('.')[0] + '.npy',np.array(photos))
        cap.release()
    return None