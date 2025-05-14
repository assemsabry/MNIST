# Neural Network for MNIST Digit Classification
This repository contains a simple implementation of a fully connected neural network built from scratch using NumPy, designed to classify handwritten digits from the MNIST dataset. The model is trained and tested on the MNIST dataset of grayscale images of size 28x28 pixels.

Project Overview
The model consists of:

An input layer that takes flattened 28x28 grayscale images (784 input features).

A hidden layer with 128 neurons using ReLU activation.

An output layer with 10 neurons using softmax activation to predict the digit class (0–9).

The training is done using mini-batch gradient descent and cross-entropy loss, implemented manually without using high-level deep learning frameworks.

Features
Manual implementation of forward and backward propagation.

Training on the MNIST dataset.

Model saving and loading using NumPy’s .npz format.

Evaluation of model accuracy on test data.

Simple prediction interface to test individual images.

Dataset
The MNIST dataset is automatically downloaded from Keras datasets. It contains:

60,000 training images.

10,000 test images.

Images are 28x28 pixels grayscale, representing digits from 0 to 9.

Requirements
Python 3.x

NumPy

Matplotlib (for visualization)

TensorFlow and Keras (for dataset loading only)

Install dependencies using pip:

bash
Copy
Edit
pip install numpy matplotlib tensorflow
Usage
Training the model
Run the training script to train the model on MNIST:

bash
Copy
Edit
python train_model.py
This will train the model for a specified number of epochs and save the weights to a file named fnn_model.npz.

Testing the model
Run the test script to load the saved model and evaluate it on the test set:

bash
Copy
Edit
python test_model.py
You can also test individual images and see predictions alongside the actual labels.

Author
Assem Sabry — Developer and Trainer of this model.

License
This project is licensed under the MIT License — see the LICENSE file for details.
