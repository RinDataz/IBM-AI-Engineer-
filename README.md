# IBM AI Engineer Specialization Repository

[**Welcome to the IBM AI Engineer Specialization repository!**](https://www.coursera.org/professional-certificates/ai-engineer)

This repository showcases my progress and projects throughout the IBM AI Engineer specialization offered on Coursera. It serves as a comprehensive portfolio of my work, encompassing various AI concepts and applications covered in the specialization.

## Contents

### Projects

Explore detailed AI projects demonstrating my skills in key areas such as natural language processing, computer vision, and predictive analytics.

#### [1. Stop Sign Detection Model](https://github.com/RinDataz/IBM-AI-Engineer-/blob/main/03-Introduction%20to%20Computer%20Vision%20and%20Image/stopsignmodel.ipynb)

**Project Summary**

- **Data Loading and Display**: Loaded and displayed 5 sample images with stop signs and 5 without using OpenCV and Matplotlib.
- **Data Augmentation**: Applied transformations using Keras' ImageDataGenerator, including rotations, shifts, rescaling, shear, zoom, horizontal flipping, and filling modes.
- **Model Architecture**: 
  - **Layers**: 
    - Convolutional layers (Conv2D) with ReLU activation
    - Max Pooling layers (MaxPooling2D)
    - Flatten layer to convert 2D matrix to a vector
    - Dense layer with ReLU activation
    - Dropout layer for regularization
    - Output layer with sigmoid activation for binary classification
  - **Compilation**: Binary cross-entropy loss, Adam optimizer, and accuracy metric.
- **Data Preparation**: Training and testing datasets loaded using ImageDataGenerator. Image dimensions set to (150, 150, 3).
- **Training the Model**: Trained for 50 epochs with accuracies and losses printed for each epoch.
- **Model Evaluation**: Achieved approximately 70% accuracy on training data. Saved as 'stop_sign_classifier.h5'.
- **Plotting Results**: Visualized training accuracy across epochs using Matplotlib.

**Key Points**

- **Data Augmentation**: Enhanced training dataset, improving model generalization.
- **CNN Architecture**: Extracted features using convolutional and pooling layers, followed by dense layers for classification.
- **Training Performance**: Addressed potential overfitting or underfitting through tuning.
- **Model Save**: Trained model saved for future use or further fine-tuning.

#### [2. Concrete Strength Prediction using Neural Networks](https://github.com/RinDataz/IBM-AI-Engineer-/blob/main/02-Introduction%20to%20Deep%20Learning%20%26%20Neural%20Networks%20with%20Keras/Concrete%20Strength%20Prediction%20using%20Neural%20Networks.ipynb)

**Project Summary**

This project predicts the compressive strength of concrete using a neural network model implemented in Keras. The notebook includes:
- **Data Preprocessing and Feature Selection**
- **Implementation of a Baseline Neural Network Model**
- **Model Evaluation**: Using mean squared error (MSE).
- **Repeated Experiments**: Computed mean and standard deviation of MSE.

[Concrete Data](https://github.com/RinDataz/IBM-AI-Engineer-/blob/main/02-Introduction%20to%20Deep%20Learning%20%26%20Neural%20Networks%20with%20Keras/concrete_data.csv)

#### [3. End Of Course Capstone Project](https://github.com/RinDataz/IBM-AI-Engineer-/tree/main/06-AI-Capstone-Project-with-Deep-Learning)

**Project Components**

- **Data Loading and Visualization**: [1.0_load_and_display_data.ipynb](link)
  - Covered methods for loading datasets and visualizing data to understand its structure and properties.

- **Data Loading with PyTorch**: [2.1_data_loader_PyTorch.ipynb](link)
  - Demonstrated efficient data loading using PyTorch's utilities, critical for feeding data into deep learning models during training and evaluation.

- **Linear Classifier Implementation**: [3.1_linearclassiferPytorch.ipynb](link)
  - Implemented a linear classifier using PyTorch, covering basics of linear models, training, and evaluation.

- **Convolutional Neural Networks (CNNs) - ResNet-18**: [4.1_resnet18_PyTorch.ipynb](link)
  - Explained implementation and use of ResNet-18 architecture, known for its residual learning framework aiding in training deep networks effectively.

These notebooks collectively cover essential deep learning techniques such as data preprocessing, model implementation, training, and evaluation using PyTorch.

## [Certification](https://github.com/RinDataz/IBM-AI-Engineer-/blob/main/IBM%20AI%20ENG%20CERT.pdf)
