import cv2
import numpy as np

def concate_images(videoPath): 
	#concate_seq= np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,10,11,12,13,14,14])
	
	frames = np.load(videoPath)

	im_1 = frames[0]
	im_2 = frames[1]
	im_3 = frames[2]
	im_4 = frames[3]
	im_5 = frames[4]
	layer1 = np.concatenate((im_1, im_2, im_3, im_4, im_5), axis=1)

	im_6 = frames[5]
	im_7 = frames[6]
	im_8 = frames[7]
	im_9 = frames[8]
	im_10 = frames[9]
	layer2 = np.concatenate((im_6, im_7, im_8, im_9, im_10), axis=1)

	im_11 = frames[10]
	im_12 = frames[11]
	im_13 = frames[12]
	im_14 = frames[13]
	im_15 = frames[14]
	layer3 = np.concatenate((im_11, im_12, im_13, im_14, im_15), axis=1)

	im_16 = frames[15]
	im_17 = frames[16]
	im_18 = frames[17]
	im_19 = frames[18]
	im_20 = frames[19]
	layer4 = np.concatenate((im_16, im_17, im_18, im_19, im_20), axis=1)

	im_21 = frames[20]
	im_22 = frames[21]
	im_23 = frames[22]
	im_24 = frames[23]
	im_25 = frames[24]
	layer5 = np.concatenate((im_21, im_22, im_23, im_24, im_25), axis=1)

	output = np.concatenate((layer1, layer2, layer3, layer4, layer5), axis=0)
	cv2.imwrite(videoPath+"concate-output.jpg", output)
	print("[INFO] concate done")