import numpy as np
import main
import enviroment as env

def get_category(id):
    return env.class_names[int(id/9)] #9 -> videos in each category

def generate_arrays(available_ids):
    from random import shuffle
	
    while True:
        
        shuffle(available_ids)
        for i in available_ids:
            
            scene = np.load('./data/{}.npy'.format(i))
            category = get_category(i)
            yield (np.array([scene]), category)