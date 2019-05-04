# Data Science: image classification
## Project 1 Term 1
Jesse Fredrickson

## Motivation
The purpose of this project is to build and train a deep neural network to classify images - specifically, I will be training it to recognize the species of a flower. The dataset, consisting of ~20,000 training images, is quite large and not stored in this repo. This project is designed to be run on a local GPU if available. It will run without one if one is not available, but user be warned it will be quite slow.

## Files
**Image Classifier Project.ipynb:** This is the main file in which I perform my analysis in python.
**Image_Classifier_Project.html:** Since the original data files for this project are not available due to storage limits, my analysis is not repeatable. This .html file contains all of the results of my work for review.
**GPUmemcheck.py:** short script to analyze current GPU memory status for debugging.
**LICENSE:** The MIT software license associated with some of the deep neural net architectures my project is capable of using
**train.py:** This file is a command line version of the training portion of the main project. It allows a user to train a model from a list of supported models on their dataset, and it produces a file called **checkpoint.pth** which contains the attributes of the trained model.
**predict.py:** This is a command line version of the prediction portion of the main project. It allows the user to use a predefined model to predict the label of a target image; in my case, the species of a given flower.
**cat_to_name.json:** A mapping of labels to names for the training dataset

## Results / Instructions
Due to storage constraints, I cannot store the training or testing data associated with this project on this repository, so my results are not explicitly repeatable by the reader. However, In this project I was able to successfully train a variety of neural networks to recognize with high accuracy the species of a flower in a given image, with a known library of 102 different species. I was able to successfully convert the project into a series of two command line programs that can be easily run on a target training and testing dataset for repeatability.

In order to run this project, you will need the torch and torchvision modules
