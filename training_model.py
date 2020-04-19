import model
import generate
import enviroment as env
from random import shuffle
# Train / Validation split

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