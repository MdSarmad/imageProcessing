#Importing keras libraries and packages

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D

#step1 Initializing CNN
classifier = Sequential()##clss is sequential and object is classifier -- object initialisation

#step2 adding 1st Convolution layer and Pooling layer
classifier.add(Convolution2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

#step3 adding 2nd convolution layer and pooling layer
classifier.add(Convolution2D(32,(3,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))


#step4 Flattening the layers
classifier.add(Flatten())

#step5 Full_Connection

classifier.add(Dense(units=32,activation = 'relu'))

classifier.add(Dense(units=64,activation = 'relu'))

classifier.add(Dense(units=128,activation = 'relu'))

classifier.add(Dense(units=256,activation = 'relu'))

classifier.add(Dense(units=256,activation = 'relu'))

classifier.add(Dense(units=6,activation = 'softmax'))

#step6 Compiling CNN
classifier.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

#step7 Fitting CNN to images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, # To rescaling the image in range of [0,1]
                                   shear_range=0.2, # To randomly shear the images
                                   zoom_range = 0.2, # To randomly zoom the images
                                   horizontal_flip = True) # for randomly flipping half of the image

test_datagen = ImageDataGenerator(rescale = 1./255)
print("\nTraining the data...\n")
training_set = train_datagen.flow_from_directory('train',
                                                target_size=(64,64),
                                                batch_size=12,
                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory('test',
                                            target_size=(64,64),
                                            batch_size=12,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch=int(1212/12), # Total training images
                         epochs = 20, # Total no. of epochs
                         validation_data = test_set,
                         validation_steps = int(300/12)) # Total testing images

#step8 saving model

classifier.save("model.h5")










