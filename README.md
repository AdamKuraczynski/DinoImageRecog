### Prereq:  

- Python 3.11.9  
- pip install tensorflow==2.13.1 --user  
- pip install keras==2.13.1 --user  
- pip install opencv-python, matplotlib, pandas, tabulate, openpyxl, seaborn, joblib  

### Features

- Hyperparameter Search: Explore various combinations of convolutional layers, pooling layers, epochs, optimizers, and batch sizes to find the optimal model configuration.
- Model Training: Train the CNN model using different hyperparameters and save the best performing model.
- Prediction Script: Load the trained model to predict the class of new dinosaur images.
- Result Analysis: Generate and save plots to visualize training accuracy against different hyperparameters.

### Directory Structure

- results/: Contains CSV files with training results and generated plots.
- trainedModel/: Stores the best performing trained model.
- dinosaursDataset/: Directory containing the training dataset.
- testingDataset/: Directory containing images for testing predictions.

### Usage

- Train the Model: Run the hyperparameter search script to train the model and save the best configuration.
- Predict: Use the prediction script to classify new dinosaur images.
- Visualize Results: Check the results/ folder for performance plots and CSV files with detailed training metrics.

### Testing variables:
- Convolutional layers 1/2/3/5
- Pooling layers 1/2/3/5
- Epochs 5/10/25/50
- Optimizer Adam/SGD/RMSpop/Adadelta
- Batch size 16/32/64/128

### Known issue - Add PATH to environmental variables ERROR
View advanced system settings -> environment variables -> click variable called "Path" -> edit -> add new line with your python scripts, example below:
C:\Users\Adam\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts  
