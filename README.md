## Project Overview

This project develops a deep learning model to predict pneumonia from chest X-ray images using computer vision techniques. The model is trained to classify X-ray images into two categories: pneumonia or normal.

## Objective

To leverage convolutional neural networks (CNN) for the detection of pneumonia in chest X-ray images.
Techniques Used
Convolutional Neural Network (CNN): Multiple convolutional layers were used for feature extraction, followed by max-pooling and dense layers.
Regularization: Dropout was applied to prevent overfitting.
Activation Functions: The final layer uses the sigmoid activation function for binary classification.

## Dataset

The dataset used for training and testing the model is publicly available on Kaggle's Chest X-ray Pneumonia Dataset. It contains X-ray images of normal and pneumonia-infected lungs, categorized into training, validation, and test sets.

## Model Architecture

Input: X-ray images resized to (150x150x3).
Layers:
Convolutional layers for feature extraction
Max-Pooling layers for dimensionality reduction
Fully connected (dense) layers for classification
Dropout for regularization
Output: Binary classification (Pneumonia or Normal).

## How to Use

Clone the repository.
Download the dataset from Kaggle and extract it into the data/ folder.
Train the model using src/training.py.
Use src/prediction.py to make predictions on new X-ray images.

## Installation

To run the project locally, first clone the repository and install the required dependencies using the following command:

```bash
pip install -r requirements.txt

Ensure you have the correct dataset structure in the data/ folder.

## Training the Model
To train the model, run the training.py script:

python src/training.py

