"""
Heart Disease Prediction - Machine Learning Classification App
A Gradio-based web application showcasing 7 different ML classification algorithms.

Author: [Your Name]
Project: IBM Machine Learning Course - Final Project
"""

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# =============================================================================
# DATA LOADING AND MODEL TRAINING
# =============================================================================

def load_and_prepare_data():
    """Load the heart disease dataset and prepare it for modeling."""
    df = pd.read_csv('heart.csv')
    df = df.drop_duplicates()

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return df, X, y, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_all_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    """Train all 7 classification models and return them with their metrics."""

    models = {}
    results = []

    # 1. Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = {'model': lr, 'scaled': True}
    results.append(evaluate_model(lr, X_test_scaled, y_test, 'Logistic Regression'))

    # 2. K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    models['K-Nearest Neighbors'] = {'model': knn, 'scaled': True}
    results.append(evaluate_model(knn, X_test_scaled, y_test, 'K-Nearest Neighbors'))

    # 3. Support Vector Machine
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    models['Support Vector Machine'] = {'model': svm, 'scaled': True}
    results.append(evaluate_model(svm, X_test_scaled, y_test, 'Support Vector Machine'))

    # 4. Decision Tree
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = {'model': dt, 'scaled': False}
    results.append(evaluate_model(dt, X_test, y_test, 'Decision Tree'))

    # 5. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = {'model': rf, 'scaled': False}
    results.append(evaluate_model(rf, X_test, y_test, 'Random Forest'))

    # 6. Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = {'model': gb, 'scaled': False}
    results.append(evaluate_model(gb, X_test, y_test, 'Gradient Boosting'))

    # 7. XGBoost
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42,
                           use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_train, y_train)
        models['XGBoost'] = {'model': xgb, 'scaled': False}
        results.append(evaluate_model(xgb, X_test, y_test, 'XGBoost'))

    results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False).reset_index(drop=True)

    return models, results_df


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and return metrics dictionary."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    }


# Load data and train models at startup
print("Loading data and training models...")
df, X, y, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_prepare_data()
models, results_df = train_all_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)
print("Models trained successfully!")

# Get feature names
feature_names = X.columns.tolist()

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, model_choice):
    """Make prediction using the selected model."""

    # Create input array
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_df = pd.DataFrame(input_data, columns=feature_names)

    # Get the selected model
    model_info = models[model_choice]
    model = model_info['model']

    # Scale if necessary
    if model_info['scaled']:
        input_processed = scaler.transform(input_df)
    else:
        input_processed = input_df.values

    # Make prediction
    prediction = model.predict(input_processed)[0]
    probability = model.predict_proba(input_processed)[0]

    # Format results
    if prediction == 1:
        result = "⚠️ HIGH RISK: Heart Disease Detected"
        result_color = "red"
    else:
        result = "✅ LOW RISK: No Heart Disease Detected"
        result_color = "green"

    prob_no_disease = probability[0] * 100
    prob_disease = probability[1] * 100

    # Create probability visualization
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.barh(['No Disease', 'Disease'], [prob_no_disease, prob_disease], color=colors, edgecolor='black')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability (%)', fontsize=12)
    ax.set_title(f'Prediction Probabilities - {model_choice}', fontsize=14, fontweight='bold')

    for bar, prob in zip(bars, [prob_no_disease, prob_disease]):
        ax.text(prob + 1, bar.get_y() + bar.get_height()/2, f'{prob:.1f}%', va='center', fontweight='bold')

    plt.tight_layout()

    details = f"""
### Prediction Details

**Model Used:** {model_choice}

**Probabilities:**
- No Heart Disease: {prob_no_disease:.2f}%
- Heart Disease: {prob_disease:.2f}%

**Input Values:**
- Age: {age} years
- Sex: {'Male' if sex == 1 else 'Female'}
- Chest Pain Type: {cp}
- Resting Blood Pressure: {trestbps} mm Hg
- Cholesterol: {chol} mg/dl
- Fasting Blood Sugar > 120: {'Yes' if fbs == 1 else 'No'}
- Resting ECG: {restecg}
- Max Heart Rate: {thalach}
- Exercise Induced Angina: {'Yes' if exang == 1 else 'No'}
- ST Depression (Oldpeak): {oldpeak}
- Slope: {slope}
- Major Vessels: {ca}
- Thalassemia: {thal}
"""

    return result, fig, details


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_model_comparison_plot():
    """Create a bar chart comparing all models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))

    for ax, metric in zip(axes.flatten(), metrics):
        sorted_df = results_df.sort_values(metric, ascending=True)
        bars = ax.barh(sorted_df['Model'], sorted_df[metric], color=colors, edgecolor='black')
        ax.set_xlabel(metric)
        ax.set_title(f'{metric} Comparison', fontweight='bold')
        ax.set_xlim(0.7, 1.0)

        for bar, val in zip(bars, sorted_df[metric]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    return fig


def create_feature_importance_plot():
    """Create feature importance plot using Random Forest."""
    rf_model = models['Random Forest']['model']

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance_df)))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors, edgecolor='black')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')

    for i, (feat, imp) in enumerate(zip(importance_df['Feature'], importance_df['Importance'])):
        ax.text(imp + 0.005, i, f'{imp:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    return fig


def create_correlation_heatmap():
    """Create correlation heatmap of features."""
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation = df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                mask=mask, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def create_target_distribution_plot():
    """Create target variable distribution plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    target_counts = df['target'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(['No Disease (0)', 'Disease (1)'], target_counts.values, color=colors, edgecolor='black')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Target Variable Distribution', fontsize=14, fontweight='bold')

    for bar, count in zip(bars, target_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(count),
                ha='center', fontweight='bold', fontsize=12)

    plt.tight_layout()
    return fig


def create_roc_curves():
    """Create ROC curves for all models."""
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for (name, model_info), color in zip(models.items(), colors):
        model = model_info['model']
        X_data = X_test_scaled if model_info['scaled'] else X_test

        y_pred_proba = model.predict_proba(X_data)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def get_dataset_info():
    """Return dataset information as formatted string."""
    info = f"""
## Dataset Overview

**Heart Disease UCI Dataset**

### Basic Statistics
- **Total Samples:** {len(df)}
- **Features:** {len(feature_names)}
- **Target Classes:** 2 (Binary Classification)
- **Missing Values:** None
- **Duplicates Removed:** Yes

### Target Distribution
- No Heart Disease (0): {(df['target'] == 0).sum()} ({(df['target'] == 0).mean()*100:.1f}%)
- Heart Disease (1): {(df['target'] == 1).sum()} ({(df['target'] == 1).mean()*100:.1f}%)

### Feature Descriptions

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Continuous |
| sex | Sex (1=male, 0=female) | Binary |
| cp | Chest pain type (0-3) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Continuous |
| chol | Serum cholesterol (mg/dl) | Continuous |
| fbs | Fasting blood sugar > 120 mg/dl | Binary |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Continuous |
| exang | Exercise induced angina | Binary |
| oldpeak | ST depression (exercise vs rest) | Continuous |
| slope | Slope of peak exercise ST segment | Categorical |
| ca | Major vessels colored by fluoroscopy | Ordinal |
| thal | Thalassemia | Categorical |
"""
    return info


def get_model_results_table():
    """Return model results as a formatted dataframe."""
    display_df = results_df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}' if x is not None else 'N/A')
    return display_df


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.gr-button-primary {
    background-color: #e74c3c !important;
}
"""

# Create the Gradio app
with gr.Blocks(css=custom_css, title="Heart Disease Prediction - ML Classification") as demo:

    gr.Markdown("""
    # ❤️ Heart Disease Prediction System
    ## Machine Learning Classification Project

    This application demonstrates **7 different machine learning algorithms** for predicting heart disease.
    Built as part of the IBM Machine Learning Course on Coursera.

    ---
    """)

    with gr.Tabs():

        # Tab 1: Prediction
        with gr.TabItem("🔮 Make Prediction"):
            gr.Markdown("### Enter Patient Information")

            with gr.Row():
                with gr.Column():
                    age = gr.Slider(20, 100, value=50, step=1, label="Age (years)")
                    sex = gr.Radio([("Male", 1), ("Female", 0)], value=1, label="Sex")
                    cp = gr.Slider(0, 3, value=0, step=1, label="Chest Pain Type (0-3)")
                    trestbps = gr.Slider(80, 200, value=120, step=1, label="Resting Blood Pressure (mm Hg)")
                    chol = gr.Slider(100, 600, value=200, step=1, label="Cholesterol (mg/dl)")
                    fbs = gr.Radio([("No", 0), ("Yes", 1)], value=0, label="Fasting Blood Sugar > 120 mg/dl")
                    restecg = gr.Slider(0, 2, value=0, step=1, label="Resting ECG Results (0-2)")

                with gr.Column():
                    thalach = gr.Slider(60, 220, value=150, step=1, label="Max Heart Rate Achieved")
                    exang = gr.Radio([("No", 0), ("Yes", 1)], value=0, label="Exercise Induced Angina")
                    oldpeak = gr.Slider(0.0, 7.0, value=1.0, step=0.1, label="ST Depression (Oldpeak)")
                    slope = gr.Slider(0, 2, value=1, step=1, label="Slope of Peak Exercise ST")
                    ca = gr.Slider(0, 4, value=0, step=1, label="Major Vessels (0-4)")
                    thal = gr.Slider(0, 3, value=2, step=1, label="Thalassemia (0-3)")

                    model_choice = gr.Dropdown(
                        choices=list(models.keys()),
                        value="Random Forest",
                        label="Select Model"
                    )

            predict_btn = gr.Button("🔍 Predict", variant="primary", size="lg")

            with gr.Row():
                with gr.Column():
                    result_text = gr.Markdown(label="Prediction Result")
                    prob_plot = gr.Plot(label="Probability Distribution")
                with gr.Column():
                    details_text = gr.Markdown(label="Details")

            predict_btn.click(
                fn=predict_heart_disease,
                inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, model_choice],
                outputs=[result_text, prob_plot, details_text]
            )

        # Tab 2: Model Comparison
        with gr.TabItem("📊 Model Comparison"):
            gr.Markdown("### Compare All 7 Machine Learning Models")

            with gr.Row():
                results_table = gr.Dataframe(
                    value=get_model_results_table(),
                    label="Model Performance Metrics",
                    interactive=False
                )

            gr.Markdown("### Performance Visualization")
            comparison_plot = gr.Plot(value=create_model_comparison_plot(), label="Model Comparison")

            gr.Markdown("### ROC Curves")
            roc_plot = gr.Plot(value=create_roc_curves(), label="ROC Curves")

        # Tab 3: Feature Analysis
        with gr.TabItem("🔬 Feature Analysis"):
            gr.Markdown("### Feature Importance & Correlation Analysis")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Feature Importance (Random Forest)")
                    importance_plot = gr.Plot(value=create_feature_importance_plot())

                with gr.Column():
                    gr.Markdown("#### Target Distribution")
                    target_plot = gr.Plot(value=create_target_distribution_plot())

            gr.Markdown("### Correlation Heatmap")
            correlation_plot = gr.Plot(value=create_correlation_heatmap())

        # Tab 4: Dataset Info
        with gr.TabItem("📋 Dataset Info"):
            gr.Markdown(get_dataset_info())

            gr.Markdown("### Sample Data")
            sample_data = gr.Dataframe(
                value=df.head(10),
                label="First 10 Rows of Dataset",
                interactive=False
            )

        # Tab 5: About
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## About This Project

            ### Overview
            This is a **Machine Learning Classification** project that predicts the presence of heart disease
            in patients based on various medical attributes. The project compares **7 different classification
            algorithms** to find the best performing model.

            ### Algorithms Implemented

            | Algorithm | Type | Key Characteristic |
            |-----------|------|-------------------|
            | **Logistic Regression** | Linear | Probability-based, interpretable |
            | **K-Nearest Neighbors** | Instance-based | Distance-based voting |
            | **Support Vector Machine** | Margin-based | Maximum margin separator |
            | **Decision Tree** | Rule-based | Hierarchical if-then rules |
            | **Random Forest** | Ensemble (Bagging) | Multiple trees, majority vote |
            | **Gradient Boosting** | Ensemble (Boosting) | Sequential error correction |
            | **XGBoost** | Ensemble (Boosting) | Regularized gradient boosting |

            ### Technical Stack
            - **Python** - Core programming language
            - **Scikit-learn** - Machine learning algorithms
            - **XGBoost** - Advanced gradient boosting
            - **Pandas & NumPy** - Data manipulation
            - **Matplotlib & Seaborn** - Visualizations
            - **Gradio** - Web interface

            ### Key Learnings
            1. Data preprocessing (scaling, handling duplicates)
            2. Train-test splitting with stratification
            3. Model training and evaluation
            4. Cross-validation for robust evaluation
            5. Hyperparameter tuning
            6. Feature importance analysis
            7. Model comparison and selection

            ### Course Information
            - **Course:** Supervised Machine Learning: Classification
            - **Provider:** IBM via Coursera
            - **Project Type:** Final Project / Portfolio Piece

            ---

            ### Contact
            - **GitHub:** [Your GitHub Profile]
            - **LinkedIn:** [Your LinkedIn Profile]
            - **Email:** [Your Email]

            ---

            *Built with ❤️ using Python and Gradio*
            """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)
