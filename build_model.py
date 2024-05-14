import numpy as np
import os
from time import time
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

path = "dinosaursDataset"
categories = os.listdir(path)
categories.sort()
print(categories)
numOfClasses = len(categories)
print("Number of categories :", numOfClasses)

# Set up data generators
path = "dinosaursDataset"
datagen = ImageDataGenerator(rescale= 1./255, validation_split=0.2, horizontal_flip=True)

train_dataset = datagen.flow_from_directory(batch_size=32,
                                            directory=path,
                                            color_mode='rgb',
                                            shuffle=True,
                                            target_size=(224,224),
                                            subset="training",
                                            class_mode="categorical")

validation_dataset = datagen.flow_from_directory(batch_size=32,
                                                 directory=path,
                                                 color_mode='rgb',
                                                 shuffle=True,
                                                 target_size=(224,224),
                                                 subset="validation",
                                                 class_mode="categorical")

# Build the model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding='same', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv2D(filters=32, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.5))

model.add(Dense(train_dataset.num_classes, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

# Training
start_time = time()

history = model.fit(train_dataset,
                    steps_per_epoch=np.ceil(train_dataset.samples / 32),
                    epochs=10,
                    validation_data=validation_dataset,
                    validation_steps=np.ceil(validation_dataset.samples / 32),
                    callbacks=[ModelCheckpoint("trainedModel/dino.h5", monitor='val_accuracy', verbose=1, save_best_only=True)])

end_time = time()
train_time = end_time - start_time

# Evaluation
train_score = model.evaluate(train_dataset)
test_score = model.evaluate(validation_dataset)

print("Train Score:", train_score)
print("Test Score:", test_score)
print("Train Time:", train_time)

num_conv_layers = sum(1 for layer in model.layers if isinstance(layer, Conv2D))
num_pooling_layers = sum(1 for layer in model.layers if isinstance(layer, MaxPooling2D))
accuracy_plot_name = f'results/{num_conv_layers}conv_{num_pooling_layers}pool_{len(history.epoch)}epochs_accuracy_plot.png'
loss_plot_name = f'results/{num_conv_layers}conv_{num_pooling_layers}pool_{len(history.epoch)}epochs_loss_plot.png'

# Plotting
plt.plot(history.history['accuracy'], 'r', label='Train accuracy')
plt.plot(history.history['val_accuracy'], 'b', label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Train and Validation Accuracy")
plt.legend(loc='lower right')
plt.savefig(accuracy_plot_name)
plt.show()

plt.plot(history.history['loss'], 'r', label='Train loss')
plt.plot(history.history['val_loss'], 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Train and Validation Loss")
plt.legend(loc='upper right')
plt.savefig(loss_plot_name)
plt.show()

summary = f"Convolutional Layers: {num_conv_layers}\n"
summary += f"Pooling Layers: {num_pooling_layers}\n"
summary += f"Number of Epochs: {len(history.epoch)}\n"
summary += "Train Loss: {}\n".format(train_score[0])
summary += "Train Accuracy: {}\n".format(train_score[1])
summary += "Test Loss: {}\n".format(test_score[0])
summary += "Test Accuracy: {}\n".format(test_score[1])
summary += f"Train Time: {train_time} seconds\n"

file_name = f"results/{num_conv_layers}conv_{num_pooling_layers}pool_{len(history.epoch)}epochs_details.txt"

with open(file_name, "w") as file:
    file.write(summary)