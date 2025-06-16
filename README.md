Clone the Repository

git clone https://github.com/Hrishi-Shekhar/Hackathon.git
cd Hackathon

Requirements-
Install required packages using:
    pip install -r requirements.txt

Pipeline Overview
1. Image Preprocessing (Resizing, normalization)

2. Feature Extraction using EfficientNetB3 + Global Average Pooling

3. Dimensionality Reduction via PCA

4. Classification using:

    Logistic Regression

    Random Forest

    SVM

    K-Nearest Neighbors

    XGBoost

5. Evaluation via:

    5-Fold Stratified Cross-Validation

    Accuracy, Precision, Recall, F1 Score (Weighted)


Performance Metrics-
For each classifier:

1. Cross-validation results

2. Final training metrics

3. Validation performance

Metrics include:

1. Accuracy

2. Precision

3. Recall

4. F1 Score



