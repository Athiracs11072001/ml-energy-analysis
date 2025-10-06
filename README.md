# Energy Consumption Prediction using Machine Learning
## Overview 

This project explores **machine learning techniques to predict electric energy consumption (Miles/kWh)**.  
It covers the full workflow — from **data cleaning and preprocessing** to **feature engineering, model training, and evaluation**.  
Three models are compared:

- Support Vector Machine (SVM)  
- Naïve Bayes  
- PyTorch Neural Network  

Performance is evaluated using **Accuracy** and **Confusion Matrix**.

---

## Dataset & Preprocessing
- Cleaned and standardized numerical variables  
- Created dummy variables for categorical features  
- Checked variance and selected relevant predictors  
- Split dataset into training and testing sets  

---

## Models Implemented
1. **Support Vector Machine (SVM)**  
   - Trained with and without dummy variables  
   - Tuned with different feature combinations  

2. **Naïve Bayes**  
   - Applied on processed categorical + numerical features  

3. **Neural Network (PyTorch)**  
   - Custom architecture built with `torch.nn`  
   - Standardized input for stable learning  

---

## Evaluation
- Metrics used: Accuracy, Confusion Matrix  
- Results visualized with plots (matplotlib & seaborn)  
- **Best Accuracy Achieved:** `0.8802 (88%)`  

| Model             | Accuracy | Notes                         |
|-------------------|----------|-------------------------------|
| SVM               | ~0.88    | Best performance overall      |
| Naïve Bayes       | lower    | Struggled with categorical    |
| Neural Network    | TBD      | Requires more tuning/epochs   |


## Tools & Libraries
- Python 3  
- pandas, numpy, matplotlib, seaborn  
- scikit-learn (SVM, Naïve Bayes, GridSearchCV)  
- PyTorch (neural network)  

## How to run

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/energy-consumption-ml.git
   cd energy-consumption-ml
2. Install dependencies: 
   pip install -r requirements.txt
   
3.Open and Run the notebook
  jupyter notebook energy-consumption-ml.ipynb

