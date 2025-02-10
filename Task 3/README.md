# Pre-training of a CNN and Fine-tuning on an Image Similarity Task  

## Overview  
This project involves pre-training a Convolutional Neural Network (CNN) on image embeddings and fine-tuning it for an image similarity task. Using a triplet-based dataset, the model learns to differentiate between similar and dissimilar images by leveraging deep feature representations.  

## Features  
- **CNN Pre-training:** Utilizes a pre-trained `VGG19` network for feature extraction.  
- **Triplet Learning:** Implements a triplet dataset structure to train a similarity-based model.  
- **Fine-tuning:** Trains a fully connected neural network on triplet embeddings.  
- **Loss Function:** Uses `Binary Cross-Entropy (BCE)` loss for optimization.  
- **Efficient Training & Testing:** Implements batch processing and GPU acceleration for training and evaluation.  

## Dependencies  
Ensure you have the required libraries installed:  
```bash
pip install torch torchvision numpy pandas pillow scikit-learn
