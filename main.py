from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_yaml
from keras.preprocessing import image
import numpy as np
import sys
import os.path


# Part 3 - Making new predictions

def predict(path):

	if os.path.exists("weights.h5") and os.path.exists("model.yaml"):
		yaml_file = open("model.yaml")
		loaded_yaml_model = yaml_file.read()
		yaml_file.close()
		classifier = model_from_yaml(loaded_yaml_model)
		classifier.load_weights("weights.h5")
		# classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])	
		print("Model loaded from file")		
	else:
		print("First use `python classifier.py` command to train the network")	
	test_image = image.load_img(path, target_size = (64, 64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = classifier.predict(test_image)

	ind = np.unravel_index(np.argmax(result), result.shape)

	classes = ["safe", "unsafe"]
	ind = ind[1]
	#if ind==0:
		#attendence for sharmi
	#else:
		#attendence for sruthy
	print(classes[ind])

if len(sys.argv)>0:
	path = str(sys.argv[1])
	predict(path)
	
