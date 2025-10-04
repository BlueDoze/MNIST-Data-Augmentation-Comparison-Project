# MNIST Data Augmentation Comparison Project

<div align="center">

![Machine Learning](https://img.shields.io/badge/Type-Machine%20Learning-blue)
![Deep Learning](https://img.shields.io/badge/Field-Deep%20Learning-orange)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-green)
![Python](https://img.shields.io/badge/Language-Python-yellow)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-red)

**A Comparative Study on the Effects of Data Augmentation in CNN Training**

</div>

## 📋 Project Overview

<div style="background: #f5f5f5; padding: 15px; border-radius: 8px; border-left: 4px solid #007acc; margin: 10px 0;">

This project demonstrates the impact of data augmentation on Convolutional Neural Network (CNN) performance using the MNIST handwritten digit dataset. Two identical CNN models were trained and evaluated - one with standard training data and another with augmented data - to analyze how data augmentation affects training dynamics, generalization capability, and final model performance.

The MNIST dataset serves as an excellent benchmark for understanding fundamental deep learning concepts. This project specifically investigates whether the additional computational cost of data augmentation translates to tangible improvements in model robustness and performance.

</div>

## 🎯 Key Findings

<div style="background: #fff3e0; padding: 15px; border-radius: 8px; border-left: 4px solid #ff9800; margin: 10px 0;">

### 📊 Performance Comparison

| Metric | No Augmentation | With Augmentation | Improvement |
|--------|-----------------|-------------------|-------------|
| **Test Accuracy** | 99.26% | 99.41% | **+0.15%** |
| **Test Loss** | 0.0288 | 0.0186 | **-35.4%** |
| **Training Time** | 663 seconds | 699 seconds | +36 seconds |

</div>

<div style="background: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 4px solid #4caf50; margin: 10px 0;">

### ⚡ Training Characteristics

- **Convergence Pattern**: Non-augmented model learned faster initially, but augmented model showed better generalization
- **Overfitting**: Non-augmented model showed larger gap between training and validation performance (0.5% vs 0.3%)
- **Learning Stability**: Augmented model demonstrated more consistent improvement across epochs

</div>

<div style="background: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196f3; margin: 10px 0;">

### 🛡️ Model Robustness

- **Better Calibration**: Lower test loss indicates more confident predictions
- **Balanced Performance**: Improved precision and recall balance across all digit classes
- **Consistent Results**: Fewer misclassifications in confusion matrix analysis

> 💡 **Insight**: Data augmentation acts as an effective regularizer, forcing the model to learn more generalized features rather than memorizing specific training examples.

</div>

## 🛠 Technical Implementation

<div style="background: #f5f5f5; padding: 15px; border-radius: 8px; border-left: 4px solid #607d8b; margin: 10px 0;">

### 📁 Dataset Specifications

| Parameter | Specification |
|-----------|---------------|
| **Dataset** | MNIST handwritten digits |
| **Training Images** | 60,000 (28×28 grayscale) |
| **Test Images** | 10,000 (28×28 grayscale) |
| **Classes** | 10 digits (0-9) |
| **Validation Split** | 20% of training data |

</div>

<div style="background: #fff3e0; padding: 15px; border-radius: 8px; border-left: 4px solid #ff9800; margin: 10px 0;">

### 🎨 Data Augmentation Techniques

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),           # ±10 degrees
    layers.RandomZoom(0.1),               # ±10% zoom
    layers.RandomTranslation(0.1, 0.1),   # ±10% horizontal/vertical shift
])
