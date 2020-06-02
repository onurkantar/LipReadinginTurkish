from generate_images_from_videos import generate_images
import model
import tensorflow as tf

#generate_images()
#validation_data,training_data = model.generate_data()
my_model = model.create_model()
my_model.load_weights("chkp/weights.hdf5")
#history = model.train_model(my_model,training_data,validation_data)
predicting_data = model.generate_predict_data()
print(predicting_data)
probabilities = my_model.predict_generator(predicting_data, 1)
print(probabilities)
#print(history)