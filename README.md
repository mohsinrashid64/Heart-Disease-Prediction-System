# Heart Disease Prediction - ML Classification Project

A machine learning web application that predicts heart disease using **7 different classification algorithms**. Built as the final project for IBM's Supervised Machine Learning: Classification course on Coursera.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-green.svg)

---

## Demo

Run locally:
```bash
python app.py
```
Then open `http://127.0.0.1:7860` in your browser.

---

## Features

| Feature | Description |
|---------|-------------|
| **Interactive Prediction** | Input patient data and get real-time predictions |
| **7 ML Algorithms** | Compare Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, XGBoost |
| **Visual Analytics** | ROC curves, feature importance, correlation heatmaps |
| **Model Comparison** | Side-by-side performance metrics for all models |

---

## Algorithms Implemented

| Algorithm | Type | Accuracy |
|-----------|------|----------|
| Logistic Regression | Linear | ~85% |
| K-Nearest Neighbors | Instance-based | ~87% |
| Support Vector Machine | Margin-based | ~88% |
| Decision Tree | Rule-based | ~82% |
| Random Forest | Ensemble (Bagging) | ~89% |
| Gradient Boosting | Ensemble (Boosting) | ~88% |
| XGBoost | Ensemble (Boosting) | ~89% |

*Accuracy values are approximate and may vary.*

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

---

## Project Structure

```
Final Project/
├── app.py                                    # Gradio web application
├── heart.csv                                 # Dataset
├── Heart_Disease_Classification_Project.ipynb # Jupyter notebook with full analysis
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

---

## Dataset

**Heart Disease UCI Dataset** - 1025 samples, 13 features

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels |
| thal | Thalassemia |

---

## Key Learnings

- Data preprocessing and feature scaling
- Train-test splitting with stratification
- Cross-validation for robust evaluation
- Hyperparameter tuning with GridSearchCV
- Model comparison and selection
- Building interactive ML applications with Gradio

---

## Tech Stack

- **Python 3.8+**
- **Scikit-learn** - ML algorithms
- **XGBoost** - Gradient boosting
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **Gradio** - Web interface

---

## Screenshots

### Prediction Interface
*Input patient data and get instant predictions with probability scores*

### Model Comparison
*Compare all 7 algorithms with performance metrics and ROC curves*

### Feature Analysis
*Visualize feature importance and correlations*

---

## License

MIT License - feel free to use this project for learning and portfolio purposes.

---

## Acknowledgments

- IBM & Coursera for the Machine Learning course
- UCI Machine Learning Repository for the dataset

---

*Built with Python and Gradio*
