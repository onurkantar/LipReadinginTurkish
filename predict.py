import numpy as np
from keras import optimizers
import model
import enviroment as env
from keras.preprocessing import image as image_utils

#örnek ["Stop navigation", "Excuse me"]
weights_path="./chkp/weights.hdf5"

# Build VGG model
my_model = model.generate_convlstm_model(90,3,256, 256,env.class_names)
# Load weights for model
my_model.load_weights(weights_path)

my_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9),
              metrics=['accuracy'])

def predict_by_model(path):
    # example path = 'val/16/M01_words_06_01result.jpg'

    print("[INFO] loading and preprocessing image...")
    input_image = load_and_prcoess_image(path)
    prediction = my_model.predict(input_image)
    prediction_class = np.argmax(prediction, axis=1)

    if(is_confidence_too_low(prediction)):
        print("Can you say again? Please")
        write_to_txt("result_lip/text.txt", "Can you say again? Please")

    else:
        print(env.class_names[prediction_class[0]])
        print(prediction_class[0]+1)
        print(prediction[0])
        write_to_txt("result_lip/text.txt", env.class_names[prediction_class[0]])

    # ID of Good Bye is 5
    if(prediction_class[0]+1==5):
        return 0
    else:
        return 1

def load_and_prcoess_image(path):
    image = image_utils.load_img(path, target_size=(175, 175))
    image = image_utils.img_to_array(image)
    input_image = np.expand_dims(image, axis=0)/255
    return input_image

def is_confidence_too_low(prediction):
    prediction_class = np.argmax(prediction, axis=1)
    return prediction[0][prediction_class[0]]<0.5

def write_to_txt(name, words):
    with open(name, "w") as text_file:
        text_file.write(words)  
