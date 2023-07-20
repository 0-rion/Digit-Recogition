# Digit_Recogition

## Introduction
This project aims to construct a neural network from scratch to recognise handwritten digits. 

## Training Data
The training data is from MNIST data set which is a large collection of handwritten digits commonly used for training machine learning models for image processing. The database contains 60,000 training images and 10,000 testing images. The images are in grayscale with size of 28 x 28, corresponding to 784 pixels. In this project, the data is seperated into 50,000 of traingin set, 10,000 of validation set and 10,000 of testing set.

## Architecture of Neural Network
This neural network uses sigmoid neuron as the fundamental building blocks. It consists of a input layer with 784 individual inputs each corresponding to the grayscale value of a pixel (from 0 to 1), a hidden layer with 15 neurons and a output layer of 10 neurons representing digits 0 - 9.
