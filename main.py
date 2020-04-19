from PIL import Image, ImageDraw
import face_recognition
import cv2
import numpy as np
import glob
import os
import model
import generate
from os import listdir
from os.path import isfile, join
import datetime as dt
import time
import enviroment as env
from random import shuffle



data_path = os.getcwd() + '\\test\\'

main_path = os.getcwd() + '\\ignoreFolder\\'

#onlyfiles = [f for f in listdir(main_path) if isfile(join(main_path, f))]

#print(onlyfiles)
## Playing video from file:
#for path in onlyfiles:
#    videoPath , videoName = path.split('_')
#    print(videoPath)
#    print(videoName)
#    innerSavePath = join(data_path,videoPath)
#    innerSavePath = innerSavePath + "\\"
#    print(innerSavePath)
#    videoPath = join(main_path, path)
#    print("video path : ")
#    print(videoPath)
#    cap = cv2.VideoCapture(videoPath)
#    try:
#        if not os.path.exists(innerSavePath):
#            os.makedirs(innerSavePath)
#    except OSError:
#        print ('Error: Creating directory of data')
#
#    face = 0
#    size = 256, 256
#    frameNumber = 1
#    photos = []
#    while(True):
#
#        ret, frame = cap.read()
#        if frame is None:
#            break
#        img = Image.fromarray(frame, 'RGB')
#        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        face_locations = face_recognition.face_locations(frame)
#        try:
#            crop_location = (face_locations[face][3],face_locations[face][0],face_locations[face][1],face_locations[face][2])
#        except IndexError:
#            print("No face in this frame.")
#            frameNumber +=1
#            continue
#        crop = img.crop(crop_location)
#
#        crop = crop.resize(size)
#        crop = np.array(crop)
#
#        img = np.zeros([256,256,3],dtype=np.uint8)
#        pil_image = Image.fromarray(img)
#        d = ImageDraw.Draw(pil_image)
#        face_landmarks_list = face_recognition.face_landmarks(crop)
#
#        for face_landmarks in face_landmarks_list:
#            print("frame {}".format(frameNumber))
#            #print(path)
#            #print("The top lip in this face has the following points: {}".format(face_landmarks["top_lip"]))   
#            #print("The bottom lip in this face has the following points: {}".format(face_landmarks["bottom_lip"]))   
#            #print("\n")
#
#
#            for facial_feature in face_landmarks.keys():
#                d.line(face_landmarks[facial_feature], width=3)
#
#            name = innerSavePath + videoName.split('.')[0] + "_" + str(frameNumber) + '.jpg'
#        #pil_image.save(name) #test (sonra yoruma al TODO)
#        photos.append(np.array(pil_image))
#        frameNumber +=1
#
#    while frameNumber <= 90 :
#        image = np.zeros([256,256,3],dtype=np.uint8)
#        photos.append(image)
#        frameNumber +=1
#
#    np.save(innerSavePath + videoName.split('.')[0] + '.npy',np.array(photos))
#    cap.release()


def train_my_model():

    available_ids = [i for i in range(1, 46)]

    print("available ids :")
    print(available_ids)
    print("shuffled !! ")
    shuffle(available_ids)
    print(available_ids)


    final_train_id = int(len(available_ids)*0.8)
    print("final_train_id")
    print(final_train_id)
    train_ids = available_ids[:final_train_id]
    print(train_ids)
    val_ids = available_ids[final_train_id:]
    print(val_ids)


    my_model = model.generate_convlstm_model(90,3,256, 256,env.class_names)

    # fit the model
    history = my_model.fit_generator(
        generate.generate_arrays(train_ids)
        , steps_per_epoch = len(train_ids)
    
        , validation_data = generate.generate_arrays(val_ids)
        , validation_steps = len(val_ids)
    
        , epochs = 100
        , verbose = 1
        , shuffle = False
        , initial_epoch = 0
        )
    return(history,model)

my_history , ML_model = train_my_model()