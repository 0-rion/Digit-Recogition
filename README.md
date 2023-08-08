# Digit Recogition with Neural Network

## Introduction
This project aims to construct a neural network library from scratch using python and test it's ability to recognise handwritten digits.

## Neural Network Library
This neural network library uses sigmoid neuron as the fundamental building blocks. NumPy library is used to improve the computational efficiency by taking advantage of matrix operations of weights and biases. 

## Architecture of Neural Network
This neural network is made up of three layers. It consists of 784 individual inputs each corresponding to the grayscale value of a pixel (from 0 to 1), a hidden layer with 15 neurons and an output layer of 10 neurons representing digits 0 - 9.

## Training Model
This model relies on stochastic gradient descend to adjust the weights and biases. Stochastic gradient descend is more computationally efficient than traditional gradient descend as gradient is computed only for a small sample of randomly chosen training data. This speeds up the process of finding gradient and thus the learning process.

## Training Data
The training data is from MNIST data set which is a large collection of handwritten digits commonly used for training machine learning models for image processing. The database contains 60,000 training images and 10,000 testing images. The images are in grayscale of size of 28 x 28, corresponding to 784 pixels. In this project, the data is seperated into 50,000 of traingin set, 10,000 of validation set and 10,000 of testing set.

# Reference
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)
