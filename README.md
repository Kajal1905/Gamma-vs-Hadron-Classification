# Gamma-vs-Hadron-Classification
Machine Learning project to classify cosmic ray events (Gamma vs Hadron) using the MAGIC Gamma Telescope dataset with Logistic Regression, Random Forest, and Neural Networks.

# Gamma vs Hadron Classification (ML Project)

## 📌 Project Overview
This project builds a machine learning model to classify cosmic events as **Gamma rays or Hadrons** using the MAGIC Gamma Telescope dataset.  
The dataset comes from a Cherenkov telescope experiment that detects cosmic rays.  
The goal is to accurately distinguish between **Gamma events (signal)** and **Hadron events (background noise)**.

---

## 🎯 Objectives
- Perform **data preprocessing** (scaling, oversampling for class balance).  
- Build **classification models** (Logistic Regression, Random Forest, Neural Networks).  
- Conduct **hyperparameter tuning** for neural networks (nodes, dropout, learning rate, batch size).  
- Evaluate models on validation data and select the best-performing one.  

---

## 📊 Dataset
- **Source**: [UCI MAGIC Gamma Telescope dataset](https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope)  
- **Size**: ~19,020 samples  
- **Features**: 10 continuous variables (such as fLength, fWidth, fSize, etc.)  
- **Target**: Binary classification  
  - `1` → Gamma (signal)  
  - `0` → Hadron (background noise)  

---

## 🧠 Models Used
- Logistic Regression  
- Random Forest Classifier  
- Neural Network (Keras/TensorFlow) with hyperparameter tuning:
  - Hidden nodes: `[16, 32, 64]`
  - Dropout: `[0, 0.2]`
  - Learning rates: `[0.01, 0.005, 0.001]`
  - Batch sizes: `[32, 64, 128]`
  - Epochs: `100`  

---

## 📈 Results
- **Best Validation Accuracy**: ~**88%** (Neural Network tuned with optimal parameters)  
- **Best Configuration**:  
  - Nodes = `32`  
  - Dropout = `0.2`  
  - Learning Rate = `0.001`  
  - Batch Size = `64`  
- The neural network outperformed Logistic Regression and was comparable to Random Forest.  
- Oversampling + feature scaling improved performance.  

---

## 🛠️ Tech Stack
- **Languages**: Python  
- **Libraries**:  
  - `pandas`, `numpy` → Data handling  
  - `matplotlib`, `seaborn` → Visualisation  
  - `scikit-learn` → Traditional ML models  
  - `tensorflow.keras` → Neural Networks  

---

## ✅ Conclusion
- Machine learning models can effectively separate **Gamma vs Hadron signals**.  
- Hyperparameter tuning significantly impacts neural network performance.  
- Future improvements:  
  - Try deeper architectures (multi-layer networks).  
  - Use ensemble methods (XGBoost, Gradient Boosting).  
  - Perform cross-validation for stronger generalisation.  

---

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/Gamma-vs-Hadron-Classification.git
   cd Gamma-vs-Hadron-Classification
