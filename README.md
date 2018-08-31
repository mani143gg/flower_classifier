# A basic Convolutional neural net implementation with keras and tensorflow

## Steps to execute

1. The dataset folder has two folders training and testing right now both are empty
2. Fill training and testing folder with images you want to classify
3. Each folder in training and testing directory will be considered as a class
For example lets say you are classifiying flowers and you have roses and lotuses, then in testing and training you should have 2 folders roses and lotuses and thats where the images will go
4. Then execute classifier.py as 
```sh
$ python classifier.py
```
5. After succesful training the model and weights of the neural nets will be saved as model.yaml and weights.h5

6. You can then classify images by executing main.py which will load the 2 files and do classification
```sh
$ python main.py [path to image]
```