# 🧵 Fashion-MNIST Classification with PyTorch  
Dataset:-
import kagglehub

# Download latest version
path = kagglehub.dataset_download("zalando-research/fashionmnist")

print("Path to dataset files:", path)

## 📌 Overview  
This project implements a deep learning model to classify images from the **Fashion-MNIST dataset** using **PyTorch**.  
The experiments focus on improving accuracy step by step using:  
- CPU vs. GPU training performance  
- Regularization techniques (Dropout, Batch Normalization, Weight Decay)  
- Hyperparameter tuning with **Optuna**  

✅ Final optimized model achieves **~95% accuracy** on the test dataset.  

---

## 🚀 Features  
- Implemented using **PyTorch**  
- **GPU acceleration** for faster training  
- Regularization: Dropout, Batch Normalization, Weight Decay  
- Hyperparameter optimization with Optuna  
- Performance improved from **83% → 95% test accuracy**  

---

## 📊 Results  

| Setup | Accuracy (Train) | Accuracy (Test) |
|-------|------------------|-----------------|
| **CPU (baseline)** | 92% | 83% |
| **GPU (larger dataset)** | 98% | 89% |
| **With Regularization** | 94% | 89% |
| **With Optuna Optimization** | 97% | **95%** |

---

## ⚙️ Hyperparameter Search Space (Optuna)  
```python
num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)
neurons_per_layer = trial.suggest_int("neurons_per_layer", 8, 128, step=8)
epochs = trial.suggest_int("epochs", 10, 100, step=10)
learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
optimizer_name = trial.suggest_categorical("optimizer", ['Adam', 'SGD', 'RMSprop'])
weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)```

🛠️ Tech Stack

Language: Python

Framework: PyTorch

Hyperparameter Optimization: Optuna

Dataset: Fashion-MNIST (Zalando’s article images, 60k training, 10k test)

