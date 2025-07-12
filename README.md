# 🧠 CIFAR10-Net-Deep-Learning-for-Image-Classification
This repository presents an image classification study using the CIFAR-10 dataset and various custom Convolutional Neural Network (CNN) architectures implemented in Keras. The goal was to design a model capable of achieving **at least 80% validation accuracy within 20 training epochs**, balancing training speed and generalization performance.

---

## 📁 Project Structure

- `notebook.ipynb` – Main notebook with all model code and results
- `requirements.txt` – Required Python libraries

---

## 🎯 Project Goal

- Achieve ≥ 80% validation accuracy within 20 epochs on CIFAR-10
- Experiment with different CNN architectures, layers, and hyperparameters
- Use the test set **only once** for final evaluation after model tuning is complete

---

## 🧪 Dataset Details

- **Dataset**: CIFAR-10 (60,000 images, 32x32 RGB, 10 classes)
- **Split**:
  - Training: 40,000 images
  - Validation: 10,000 images
  - Test: 10,000 images
- **Source**: Loaded directly via `keras.datasets.cifar10.load_data()`

---

## 🔧 Requirements

Install the required libraries with:

```bash
pip install -r requirements.txt
```

---

## 🧬 Model Experiments

Experimented with 10 different CNN architectures. Below is a summary of each:


### **Model 1 – Baseline CNN with RMSprop**
- Simple CNN with two convolutional blocks followed by dense layers
- Used a mix of MaxPooling and AveragePooling; optimizer: RMSprop
- **Validation Accuracy:** ~73% (after 10 epochs)

---

### **Model 2 – Deeper CNN with Adam Optimizer**
- Added an extra convolutional block for increased depth (6 Conv2D layers in total)
- Used only MaxPooling layers and switched optimizer to Adam for better convergence
- **Validation Accuracy:** ~70% (after 20 epochs)

---

### **Model 3 – Transfer Learning with VGG16**
- Used pretrained VGG16 (without top layers) as a frozen feature extractor
- Added custom dense classifier on top; input preprocessed using vgg16.preprocess_input
- **Validation Accuracy:** ~32% (after 20 epochs)

---

### **Model 4 – Transfer Learning with ResNet50**
- Used pretrained ResNet50 (excluding top layers) as a frozen feature extractor  
- Applied ResNet-specific preprocessing; added dense layers for classification  
- **Validation Accuracy**: ~36% (after 20 epochs)

---

### **Model 5 – Hyperparameter Tuned CNN (Keras Tuner)**
- Used Keras Tuner with Random Search to explore Conv2D layers, filter sizes, dense units, optimizers, and learning rates
- Included two convolutional blocks and multiple dense layers with dynamic depth and width
- **Best Validation Accuracy:** ~74.5% (did not meet 80% target within 20 epochs)

---

### **Model 6 – Advanced Hyperparameter Tuning with Custom Pooling**
- Used Keras Tuner with Random Search to optimize convolutional layers, filter sizes, dense units, optimizers, learning rate, and **pooling strategy (avg vs max)**  
- Custom dynamic architecture with up to 4 Conv2D layers per block and 6 dense layers; batch size reduced to 8 for finer updates  
- **Best Validation Accuracy**: ~73.2% (after 20 epochs across 10 trials)  

---

### **Model 7 – Deep CNN with L2 Regularization**
- Designed a deeper CNN with 3 convolutional blocks and dense layers, all using L2 regularization to reduce overfitting  
- Used high filter sizes (up to 512) and multiple dense layers with regularization  
- **Validation Accuracy**: ~0.09% (after 20 epochs)

---

### **Model 8 – Deep CNN with Batch Normalization and Dropout**
- Built on Model 7 with additional **Batch Normalization** after each Conv2D layer and **Dropout** after each pooling layer  
- Combined L2 regularization, normalization, and dropout to improve generalization and reduce overfitting  
- **Validation Accuracy**: ~80.2% (after 20 epochs)

---

## 📈 Results

| Network | No of Conv Layers | No of Filters/Layer | Pooling | No of Dense Layers| No. Neurons/Layer | Batch Size | No. Epochs | Dropout (Y/N) | BatchNormalization (Y/N) | Trainable Parameters | Non Trainable Parameters | Pretrained Model | Kernel Regularizer | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss | 
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:| :-:|:-:| :-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 4 | 32/32/32/32 | Max/Avg | 2 | 128/128 | 32 | 10 | N | N | 308,714 | - | - | - | 0.8646 | 0.7330 | 0.3950 | 1.0201 |
| 2 | 6 | 32/32/32/32/32/32 | Max/Max/Max | 2 | 128/128 | 16 | 20 | N | N | 130,602 | - | - | - | 0.8604 | 0.7089 | 0.3877 | 1.0558 |
| 3 | - | - | - | 2 | 128/128 | 16 | 20 | N | N | 83,466 | 14,714,688 | VGG16 | - | 0.3249 | 0.3207 | 1.8375 | 1.8607 |
| 4 | - | - | - | 2 | 128/128 | 16 | 20 | N | N | 280,074 | 23,587,712 | ResNet50 | - | 0.3755 | 0.3637 | 1.7329 | 1.7478  |
| 5 | 2 | 128/256 | Max/Max | 3 | 384/160/160 | 16 | 20 | N | N | 6,679,562 | - | - | - | 0.9808 | 0.7336 | 0.0662 | 2.1418 |
| 6 | 4 | 160/32/64/256 | Avg/Avg | 1 | 192 | 8 | 20 | N | N | 3,634,474 | - | - | - | 0.9855 | 0.7211 | 0.0448 | 1.9907 |
| 7 | 6 | 128/128/256/256/512/512 | Max/Max/Max | 4 | 512/512/256/256 | 8 | 20 | N | N | 9,233,546 | - | - | l2(1e-4) | 0.0979 | 0.0952 | 2.3029 | 2.3029 |
| 8 | 6 | 128/128/256/256/512/512 | Max/Max/Max | 4 | 512/512/256/256 | 8 | 20 | Y(0.2)/Y(0.3) | Y(6) | 9,237,130 | - | - | l2(1e-4) | 0.8454 | 0.8029 | 0.8884 | 1.0342 |


## ✅ Final Evaluation

The **final model (Model 8)** was evaluated on the test set after model tuning. It achieved:

- **Test Accuracy**: *~79.95%*

---

