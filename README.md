# Medical MNIST Classification with Few-Shot Learning

# This project implements a few-shot learning approach using a Prototypical Network (ProtoNet) for classifying medical images from the Medical MNIST dataset. It also includes a baseline Convolutional Neural Network (CNN) for comparison. The code is written in Python using PyTorch and includes data visualization, model training, evaluation, and t-SNE embedding analysis.

# Table of Contents





# Project Overview



# Dataset



# Features



# Requirements



# Installation



# Usage



# File Structure



# Results



# Contributing



# License

# Project Overview

# The goal of this project is to classify medical images from the Medical MNIST dataset, which contains 64x64 grayscale images across six classes: AbdomenCT, BreastMRI, ChestCT, CXR, Hand, and HeadCT. The project compares two approaches:





# Prototypical Network (ProtoNet): A few-shot learning model that learns to classify images with limited examples (n-way, k-shot).



# Baseline CNN: A standard convolutional neural network trained with supervised learning for comparison.

# The project includes data preprocessing, model training, evaluation metrics (accuracy, precision, recall, F1-score), confusion matrix visualization, and t-SNE visualization of learned embeddings.

# Dataset

# The Medical MNIST dataset consists of 64x64 grayscale medical images across six classes. The dataset is expected to be organized in the following structure:

# data_dir/
# ├── AbdomenCT/
# ├── BreastMRI/
# ├── ChestCT/
# ├── CXR/
# ├── Hand/
# ├── HeadCT/

# Each folder contains images for the respective class. You can download the dataset or use your own compatible dataset.

# Features





# Custom Dataset Class: Loads and preprocesses Medical MNIST images using PyTorch's Dataset class.



# Data Visualization: Displays representative images from each class.



# ProtoNet Implementation: Few-shot learning with customizable n-way, k-shot, and q-query settings.



# Baseline CNN: A standard CNN for comparison with ProtoNet.



# Training and Evaluation: Training loops for both models, with loss plotting and evaluation metrics.



# t-SNE Visualization: Visualizes learned embeddings in 2D space to analyze model performance.



# Confusion Matrix: Visualizes classification performance for the few-shot model.

# Requirements

# To run this project, you need the following dependencies:





# Python 3.8+



# PyTorch



# torchvision



# NumPy



# Pillow (PIL)



# scikit-learn



# matplotlib



# seaborn



# tqdm



# Jupyter Notebook (optional, for interactive execution)

# Installation





# Clone the repository:

# git clone https://github.com/your-username/medical-mnist-few-shot.git
# cd medical-mnist-few-shot



# Create a virtual environment (optional but recommended):

# python -m venv venv
# source venv/bin/activate  # On Windows: venv\Scripts\activate



# Install the required packages:

# pip install torch torchvision numpy pillow scikit-learn matplotlib seaborn tqdm



# Ensure the Medical MNIST dataset is placed in the appropriate directory (e.g., C:\Users\sonke\Desktop\python CA2 or update the data_dir path in the code).

# Usage





# Update the data_dir variable in the script to point to your dataset directory:

# data_dir = r'path/to/your/medical_mnist_dataset'



# Run the script or Jupyter Notebook:

# python medical_mnist_few_shot.py

# or open the notebook in Jupyter:

# jupyter notebook medical_mnist_few_shot.ipynb



# The script will:





# Load and preprocess the dataset.



# Visualize representative images.



# Train the baseline CNN and evaluate its accuracy.



# Train the ProtoNet model and evaluate its performance (accuracy, precision, recall, F1-score).



# Generate visualizations (loss curves, confusion matrix, t-SNE embeddings).

# File Structure

# medical-mnist-few-shot/
# ├── medical_mnist_few_shot.py  # Main Python script (or .ipynb for Jupyter)
# ├── README.md                  # This file
# ├── data/                      # Dataset directory (not included, user-provided)
# │   ├── AbdomenCT/
# │   ├── BreastMRI/
# │   └── ...
# └── outputs/                   # Generated plots (e.g., loss curves, t-SNE, confusion matrix)

# Results





# Baseline CNN: Achieves classification accuracy on the test set (reported during execution).



# ProtoNet: Evaluated on few-shot tasks (e.g., 5-way, 5-shot) with metrics including accuracy, precision, recall, and F1-score.



# Visualizations:





# Representative images for each class.



# Training loss curves for both models.



# Confusion matrix for ProtoNet.



# t-SNE visualization of ProtoNet embeddings.

# Sample output:

# Baseline CNN Accuracy: 0.XXXX
# Training Prototypical Network...
# Epoch [20/20], Loss: X.XXXX
# Evaluating on test set...
# Accuracy: 0.XXXX, Precision: 0.XXXX, Recall: 0.XXXX, F1-Score: 0.XXXX

# Contributing

# Contributions are welcome! Please follow these steps:





# Fork the repository.



# Create a new branch (git checkout -b feature/your-feature).



# Commit your changes (git commit -m 'Add your feature').



# Push to the branch (git push origin feature/your-feature).



# Open a Pull Request.

# License

# This project is licensed under the MIT License. See the LICENSE file for details.
