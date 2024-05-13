import tensorflow as tf
import os
from keras.utils import img_to_array, load_img
import numpy as np
import cv2

model = tf.keras.models.load_model("trainedModel/dino.h5")  
print(model.summary())

batchSize = 32

source_folder = "dinosaursDataset"
categories = os.listdir(source_folder)
categories.sort()
print(categories)

numOfClasses = len(categories)
print("Number of categories :")
print(numOfClasses)

def prepareImage(pathForImage):
    image = load_img(pathForImage, target_size=(224,224))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult, axis=0)
    imgResult = imgResult /255.
    return imgResult

testImagPath = "testingDataset/trice_test.png" 
imageForModel = prepareImage(testImagPath)

resultArray = model.predict(imageForModel, batch_size=batchSize, verbose=1)

answers = np.argmax(resultArray, axis=1)

print(answers[0])

text = categories[answers[0]]  
print ("Predicted : "+ text)  

img = cv2.imread(testImagPath)  
font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img, text, (0,20), font, 0.5, (209,19,77), 2)
cv2.imshow('img', img)  
cv2.waitKey(0)

cv2.destroyAllWindows()