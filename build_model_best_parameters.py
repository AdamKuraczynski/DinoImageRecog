import numpy as np
import os
from time import time
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tabulate import tabulate
import pandas as pd

# Define the parameters to test
conv_layers = [2]
pool_layers = [3]
epochs = [50]
optimizers = ['Adam']
batch_sizes = [16]

# Initialize results list
results = []

# Set up data generators
path = "dinosaursDataset"
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True)

# Function to build and train the model
def build_and_train_model(num_conv_layers, num_pool_layers, num_epochs, optimizer, batch_size):
    train_dataset = datagen.flow_from_directory(batch_size=batch_size,
                                                directory=path,
                                                color_mode='rgb',
                                                shuffle=True,
                                                target_size=(224, 224),
                                                subset="training",
                                                class_mode="categorical")

    validation_dataset = datagen.flow_from_directory(batch_size=batch_size,
                                                     directory=path,
                                                     color_mode='rgb',
                                                     shuffle=True,
                                                     target_size=(224, 224),
                                                     subset="validation",
                                                     class_mode="categorical")

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding='same', input_shape=(224, 224, 3)))
    for _ in range(num_conv_layers - 1):
        model.add(Conv2D(filters=32, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    for _ in range(num_pool_layers - 1):
        model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(train_dataset.num_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    #early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    
    start_time = time()
    history = model.fit(train_dataset,
                        steps_per_epoch=np.ceil(train_dataset.samples / batch_size),
                        epochs=num_epochs,
                        validation_data=validation_dataset,
                        validation_steps=np.ceil(validation_dataset.samples / batch_size),
                        callbacks=[ModelCheckpoint("trainedModel/dino.h5", monitor='val_accuracy', verbose=1, save_best_only=True), early_stopping ])
    end_time = time()
    train_time = end_time - start_time

    train_score = model.evaluate(train_dataset)
    test_score = model.evaluate(validation_dataset)

    results.append([num_conv_layers, num_pool_layers, num_epochs, optimizer, batch_size,
                    train_score[0], train_score[1], test_score[0], test_score[1], train_time])

# Iterate over all parameter combinations
for conv_layer in conv_layers:
    for pool_layer in pool_layers:
        for num_epochs in epochs:
            for optimizer in optimizers:
                for batch_size in batch_sizes:
                    print(f"Testing with conv_layers={conv_layer}, pool_layers={pool_layer}, epochs={num_epochs}, optimizer={optimizer}, batch_size={batch_size}")
                    build_and_train_model(conv_layer, pool_layer, num_epochs, optimizer, batch_size)

# Print results in a table
headers = ["Conv Layers", "Pool Layers", "Epochs", "Optimizer", "Batch Size", "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy", "Train Time (s)"]
print(tabulate(results, headers=headers))

# Save results to CSV
results_df = pd.DataFrame(results, columns=headers)
results_df.to_csv("results/results_best_parameters.csv", index=False)
print("Results saved to results/results_best_parameters.csv")