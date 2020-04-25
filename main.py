from generate_images_from_videos import generate_images
import model


#generate_images()
validation_data,training_data = model.generate_data()
my_model = model.create_model()
history = model.train_model(my_model,training_data,validation_data)