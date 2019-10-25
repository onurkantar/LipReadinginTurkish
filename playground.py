import dlib
import face_recognition
image = face_recognition.load_image_file("test_image.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)