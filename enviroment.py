import os
import dlib

class_names=["Günaydın","Corona","Merhaba","Selam","Teşekkürler"]

data_path = os.getcwd() + '\\data\\'

main_path = os.getcwd() + '\\ignoreFolder\\'

detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file


