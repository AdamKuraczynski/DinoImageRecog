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

# Fixed parameters
num_conv_layers = 1
num_pool_layers = 5
num_epochs = 25
optimizer = 'RMSprop'
batch_size = 64

# Parameters to test, pool sizes and strides lowered to reduce change for too much of downzising
filters = [16, 32, 64]
kernel_sizes = [(3,3), (5,5)]
activations = ['relu', 'tanh']
paddings = ['same', 'valid']
pool_sizes = [(2,2)]
strides = [(2,2)]
kernel_initializers = ['he_normal', 'glorot_uniform']

results = []

path = "dinosaursDataset"
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True)

def build_and_train_model(filter_size, kernel_size, activation, padding, pool_size, stride, kernel_initializer):
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
    model.add(Conv2D(filters=filter_size, kernel_size=kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding, input_shape=(224, 224, 3)))
    
    for _ in range(num_conv_layers - 1):
        model.add(Conv2D(filters=filter_size, kernel_size=kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding))
    
    for _ in range(num_pool_layers):
        model.add(MaxPooling2D(pool_size=pool_size, strides=stride, padding=padding))
    
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=kernel_initializer))
    model.add(Dropout(0.5))
    model.add(Dense(train_dataset.num_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

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

    results.append([filter_size, kernel_size, activation, padding, pool_size, stride, kernel_initializer,
                    train_score[0], train_score[1], test_score[0], test_score[1], train_time])

for filter_size in filters:
    for kernel_size in kernel_sizes:
        for activation in activations:
            for padding in paddings:
                for pool_size in pool_sizes:
                    for stride in strides:
                        for kernel_initializer in kernel_initializers:
                            print(f"Testing with filter_size={filter_size}, kernel_size={kernel_size}, activation={activation}, padding={padding}, pool_size={pool_size}, stride={stride}, kernel_initializer={kernel_initializer}")
                            build_and_train_model(filter_size, kernel_size, activation, padding, pool_size, stride, kernel_initializer)

headers = ["Filters", "Kernel Size", "Activation", "Padding", "Pool Size", "Stride", "Kernel Initializer", "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy", "Train Time (s)"]
print(tabulate(results, headers=headers))

results_df = pd.DataFrame(results, columns=headers)
results_df.to_csv("results/results_best_parameters.csv", index=False)
print("Results saved to results/results_best_parameters.csv")