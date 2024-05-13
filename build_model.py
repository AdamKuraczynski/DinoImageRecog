import numpy as np  # Importing NumPy library and aliasing it as 'np'
import cv2  # Importing OpenCV library for computer vision tasks
import os  # Importing os module for interacting with the operating system

path = "C:/temp/DataDinousaurRecog/Dinosaurs"  # Assigning the path to the dataset directory to the variable 'path'
categories = os.listdir(path)  # Listing all directories and files in the specified path and storing them in 'categories'
categories.sort()  # Sorting the list of categories alphabetically
print(categories)  # Printing the list of categories

numOfClasses = len(categories)  # Getting the number of categories and storing it in 'numOfClasses'
print("Number of categories :")  # Printing a message
print(numOfClasses)  # Printing the number of categories

########################################################
batchSize = 32  # Setting the batch size to 32
########################################################
imageSize = (224,224)  # Defining the target size for images to (224, 224)

# Prepare the data
from keras.preprocessing.image import ImageDataGenerator  # Importing ImageDataGenerator from Keras

datagen = ImageDataGenerator(rescale= 1./255, validation_split=0.2, horizontal_flip=True)  # Creating an ImageDataGenerator object with specified transformations

train_dataset = datagen.flow_from_directory(batch_size=batchSize,  # Generating a data flow for training set from directory with specified parameters
                                           directory=path,
                                           color_mode='rgb',
                                           shuffle=True,
                                           target_size=imageSize,
                                           subset="training",
                                           class_mode="categorical")

validation_dataset = datagen.flow_from_directory(batch_size=batchSize,  # Generating a data flow for validation set from directory with specified parameters
                                           directory=path,
                                           color_mode='rgb',
                                           shuffle=True,
                                           target_size=imageSize,
                                           subset="validation",
                                           class_mode="categorical")

batch_x, batch_y = next(train_dataset)  # Retrieving the next batch of data and labels from the training dataset

# Print the shapes of the first patch (images and labels)
print('Batch of images shape ', batch_x.shape)  # Printing the shape of batch of images
print('Batch of images label ', batch_y.shape)  # Printing the shape of batch of labels

# Build the CNN model
from keras.models import Sequential  # Importing Sequential model from Keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense  # Importing various layers from Keras

model = Sequential()  # Creating a Sequential model
model.add(Conv2D(filters=16, kernel_size =3, activation='relu', padding='same', input_shape=(224,224,3)))  # Adding a convolutional layer to the model
model.add(MaxPooling2D(pool_size=2))  # Adding a max pooling layer
model.add(Dropout(0.5))  # Adding a dropout layer

model.add(Conv2D(filters=32, kernel_size =3, activation='relu', padding='same'))  # Adding another convolutional layer
model.add(MaxPooling2D(pool_size=2))  # Adding another max pooling layer

model.add(Conv2D(filters=64, kernel_size =3, activation='relu', padding='same'))  # Adding another convolutional layer
model.add(MaxPooling2D(pool_size=2))  # Adding another max pooling layer
model.add(Dropout(0.5))  # Adding another dropout layer

model.add(Flatten())  # Flattening the output from the convolutional layers

model.add(Dense(128, activation='relu'))  # Adding a fully connected layer
model.add(Dropout(0.5))  # Adding a dropout layer

model.add(Dense(numOfClasses, activation='softmax'))  # Adding a softmax output layer with number of classes as output units

print(model.summary())  # Printing the summary of the model architecture

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])  # Compiling the model with specified loss function, optimizer, and metrics

stepsPerEpochs = np.ceil(train_dataset.samples / batchSize)  # Calculating the number of steps per epoch for training
validationSteps = np.ceil(validation_dataset.samples / batchSize)  # Calculating the number of validation steps

# Early stop
from keras.callbacks import ModelCheckpoint  # Importing ModelCheckpoint callback from Keras

best_model_file= "c/temp/dino.h5"  # Defining the file path to save the best model
best_model = ModelCheckpoint(best_model_file, monitor='val_accuracy', verbose=1, save_best_only=True)  # Creating a callback to save the best model during training

history = model.fit(train_dataset,  # Training the model with specified parameters
                    steps_per_epoch=stepsPerEpochs,
                    epochs=50,
                    validation_data=validation_dataset,
                    validation_steps=validationSteps,
                    callbacks=[best_model] )

import matplotlib.pyplot as plt  # Importing matplotlib for plotting

acc = history.history['accuracy']  # Getting the accuracy values from training history
val_acc = history.history['val_accuracy']  # Getting the validation accuracy values from training history
loss = history.history['loss']  # Getting the loss values from training history
val_loss = history.history['val_loss']  # Getting the validation loss values from training history

epochsForGraph = range(len(acc))  # Generating a range of epochs for plotting

# Plot the train and validation
plt.plot(epochsForGraph, acc, 'r', label='Train accuracy')  # Plotting training accuracy
plt.plot(epochsForGraph, val_acc, 'b', label='Validation accuracy')  # Plotting validation accuracy
plt.xlabel('Epochs')  # Labeling x-axis
plt.ylabel('Accuracy')  # Labeling y-axis
plt.title("Train and Validation Accuracy")  # Setting the title
plt.legend(loc='lower right')  # Adding legend
plt.show()  # Displaying the plot

# Plot loss and validation loss
plt.plot(epochsForGraph, loss, 'r', label='Train loss')  # Plotting training loss
plt.plot(epochsForGraph, val_loss, 'b', label='Validation loss')  # Plotting validation loss
plt.xlabel('Epochs')  # Labeling x-axis
plt.ylabel('Loss')  # Labeling y-axis
plt.title("Train and Validation Loss")  # Setting the title
plt.legend(loc='upper right')  # Adding legend
plt.show()  # Displaying the plot
