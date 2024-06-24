# IBM AI Engineer Specialization Repository


**Welcome to the IBM AI Engineer Specialization repository!**

This repository is dedicated to showcasing my progress and projects throughout the IBM AI Engineer specialization offered on Coursera. It serves as a comprehensive portfolio of my work, encompassing various AI concepts and applications covered in the specialization.

**Contents**

**Projects**

Explore detailed AI projects demonstrating my skills in key areas such as natural language processing, computer vision, and predictive analytics.

[1- Stop Sign Detection Model](https://github.com/RinDataz/IBM-AI-Engineer-/blob/main/03-Introduction%20to%20Computer%20Vision%20and%20Image/stopsignmodel.ipynb)

**Project Summary**

- Data Loading and Display:

5 sample images with stop signs and 5 sample images without stop signs are loaded and displayed using OpenCV and Matplotlib.
Data Augmentation:

Defined parameters for data augmentation using Keras' ImageDataGenerator. This includes rotations, shifts, rescaling, shear, zoom, horizontal flipping, and filling modes.
Model Architecture:

A Convolutional Neural Network (CNN) is created using Keras Sequential API.

- Layers:

Convolutional layers (Conv2D) with ReLU activation

Max Pooling layers (MaxPooling2D)

Flatten layer to convert 2D matrix to a vector

Dense layer with ReLU activation

Dropout layer for regularization

Output layer with sigmoid activation for binary classification

The model is compiled with binary cross-entropy loss, Adam optimizer, and accuracy metric.

- Data Preparation:

Training and testing datasets are loaded from directories using ImageDataGenerator.
Image dimensions are set to (150, 150, 3).

- Training the Model:

The model is trained for 50 epochs.
Training and validation accuracies and losses are printed for each epoch.

- Model Evaluation:

The final model accuracy on the training data is approximately 70%.
The model is saved as 'stop_sign_classifier.h5'.

- Plotting Results:

The training accuracy across epochs is plotted using Matplotlib.

- Key Points:
  
Data Augmentation: Enhances the training dataset by applying random transformations, improving the model's ability to generalize.

CNN Architecture: Utilizes multiple convolutional and pooling layers to extract features, followed by dense layers for classification.

Training Performance: Shows fluctuations in accuracy and loss, indicating potential overfitting or underfitting issues that could be addressed with further tuning.

Model Save: The trained model is saved for future use or further fine-tuning.

[2-Concrete Strength Prediction using Neural Networks](https://github.com/RinDataz/IBM-AI-Engineer-/blob/main/02-Introduction%20to%20Deep%20Learning%20%26%20Neural%20Networks%20with%20Keras/Concrete%20Strength%20Prediction%20using%20Neural%20Networks.ipynb)

**Project Summary** 

This project aims to predict the compressive strength of concrete using a neural network model implemented in Keras. The notebook includes:
Data preprocessing and feature selection.
Implementation of a baseline neural network model.
Model evaluation using mean squared error (MSE).
Repeated experiments to compute the mean and standard deviation of MSE.

[Concerete Data](https://github.com/RinDataz/IBM-AI-Engineer-/blob/main/02-Introduction%20to%20Deep%20Learning%20%26%20Neural%20Networks%20with%20Keras/concrete_data.csv)

-------

[3-End Of Course capstone project](https://github.com/RinDataz/IBM-AI-Engineer-/tree/main/06-AI-Capstone-Project-with-Deep-Learning)

**Data Loading and Visualization:**

Notebook: 1.0_load_and_display_data.ipynb

Technique: This notebook covers methods for loading datasets and visualizing data to understand its structure and properties.

**Data Loading with PyTorch:**

Notebook: 2.1_data_loader_PyTorch.ipynb

Technique: This notebook demonstrate how to efficiently load data using PyTorch's data loader utilities, which are critical for feeding data into deep learning models during training and evaluation.

**Linear Classifier Implementation:**

Notebook: 3.1_linearclassiferPytorch.ipynb

Technique: This notebook focus on implementing a linear classifier using PyTorch, covering the basics of linear models and how to train and evaluate them.

**Convolutional Neural Networks (CNNs) - ResNet-18:**

Notebook: 4.1_resnet18_PyTorch.ipynb

Technique: This notebook explains the implementation and use of the ResNet-18 architecture, a deep convolutional neural network known for its residual learning framework, which helps in training very deep networks effectively.

These notebooks collectively cover essential deep learning techniques such as data preprocessing, model implementation, training, and evaluation using PyTorch
