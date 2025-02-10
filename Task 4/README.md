# Transfer Learning with an MLP for Molecule Properties Regression  

## Overview  
This project applies **transfer learning** with a **Multilayer Perceptron (MLP)** to predict molecular properties using regression.  
The model is first **pre-trained** on a larger dataset and then **fine-tuned** on a smaller, target dataset. The pre-trained layers are frozen during fine-tuning, and only the final regressor is retrained.  

## Features  
- **Pre-training on a larger dataset** to learn useful representations.  
- **Fine-tuning** by freezing early layers and retraining the final regressor.  
- **Efficient training and testing** using PyTorch and `DataLoader`.  
- **Mean Squared Error (MSE) loss** for optimizing regression performance.  

## Dependencies  
Ensure you have the required dependencies installed before running the project:  
```bash
pip install torch torchvision numpy pandas
