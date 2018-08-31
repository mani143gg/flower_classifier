from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import model_from_yaml


classifier = Sequential()

classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.1))


classifier.add(Conv2D(64, (3,3), input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))


classifier.add(Conv2D(128, (3,3), input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))

classifier.add(Conv2D(256, (3,3), input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))


classifier.add(Flatten())


# classifier.add(Dense(units=128, activation='relu'))


classifier.add(Dense(units=2, activation='softmax'))


import os.path

if os.path.exists("weights.h5") and os.path.exists("model.yaml"):
	yaml_file = open("model.yaml")
	loaded_yaml_model = yaml_file.read()
	yaml_file.close()
	classifier = model_from_yaml(loaded_yaml_model)
	classifier.load_weights("weights.h5")
	# classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])	
	print("Model loaded from file")
else:
	classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
	from keras.preprocessing.image import ImageDataGenerator
	train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)

	test_datagen = ImageDataGenerator(rescale = 1./255)

	training_set = train_datagen.flow_from_directory('dataset/testing',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
	test_set = test_datagen.flow_from_directory('dataset/training',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
	classifier.fit_generator(training_set,steps_per_epoch = 4000,epochs = 20,validation_data = test_set,validation_steps = 2000)	


	model_yaml = classifier.to_yaml()
	
	with open("model.yaml", "w") as yaml_file:
		yaml_file.write(model_yaml)

	classifier.save_weights("weights.h5")
	print("Model saved to file")

# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/test.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

classes = ["safe", "unsafe"]
result = result[0]
ind = np.where(result==1)[0]
print(classes[ind[0]])