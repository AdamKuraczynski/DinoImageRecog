import tensorflow as tf  # Importing TensorFlow library
import os  # Importing os module for interacting with the operating system
from keras.utils import img_to_array, load_img  # Importing utility functions from Keras for image conversion
import numpy as np  # Importing NumPy library
import cv2  # Importing OpenCV library for computer vision tasks

model = tf.keras.models.load_model("c/temp/dino.h5")  # Loading the trained model from the specified path
print(model.summary())  # Printing the summary of the loaded model architecture

########################################################
batchSize = 32  # Setting the batch size for prediction
########################################################

# categories
source_folder = "C:/temp/DataDinousaurRecog/Dinosaurs"  # Defining the path to the folder containing categories (dinosaur images)
categories = os.listdir(source_folder)  # Listing all directories and files in the specified folder
categories.sort()  # Sorting the list of categories alphabetically
print(categories)  # Printing the list of categories

numOfClasses = len(categories)  # Getting the number of categories and storing it in 'numOfClasses'
print("Number of categories :")  # Printing a message
print(numOfClasses)  # Printing the number of categories

def prepareImage(pathForImage):
    image = load_img(pathForImage, target_size=(224,224))  # Loading the image from the specified path and resizing it to (224,224)
    imgResult = img_to_array(image)  # Converting the image to a NumPy array
    imgResult = np.expand_dims(imgResult, axis=0)  # Adding an extra dimension to the array
    imgResult = imgResult /255.  # Normalizing the pixel values
    return imgResult  # Returning the preprocessed image

# run the prediction
testImagPath = "C:/temp/DataDinousaurRecog/ptero_test.png"  # Defining the path to the test image
imageForModel = prepareImage(testImagPath)  # Preparing the test image for prediction

resultArray = model.predict(imageForModel, batch_size=batchSize, verbose=1)  # Making a prediction on the test image

########################################################
answers = np.argmax(resultArray, axis=1)  # Finding the index of the category with the highest probability
########################################################

print(answers[0])  # Printing the index of the predicted category

text = categories[answers[0]]  # Getting the predicted category label based on the index
print ("Predicted : "+ text)  # Printing the predicted category label

# the image
img = cv2.imread(testImagPath)  # Loading the test image using OpenCV
font = cv2.FONT_HERSHEY_COMPLEX  # Choosing a font for text overlay

cv2.putText(img, text, (0,20), font, 0.5, (209,19,77), 2)  # Adding text overlay with the predicted category label
cv2.imshow('img', img)  # Displaying the image with text overlay
cv2.waitKey(0)  # Waiting for a key press to exit

cv2.destroyAllWindows()  # Closing all OpenCV windows
