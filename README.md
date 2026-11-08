# Machine Learning Classification Dashboard

**BITS Pilani – M.Tech (Data Science & Engineering)**  
**Machine Learning – Assignment 2**

## 1. Problem Statement

This project implements and compares five supervised classification algorithms on the same dataset, evaluates them using six metrics, and demonstrates their predictions in an interactive Streamlit application.

Models: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM).

## 2. Dataset Description

**Dataset:** Wisconsin Diagnostic Breast Cancer  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (loaded using `sklearn.datasets.load_breast_cancer`)

| Property | Value |
|---|---|
| Instances | 569 |
| Numerical input features | 30 |
| Malignant (target 0) | 212 |
| Benign (target 1) | 357 |
| Missing values | 0 |
| Task | Binary classification |

An 80:20 stratified train/test split (`random_state=42`) yielded 455 training and 114 test records. Logistic Regression, KNN, and SVM use `StandardScaler` in training pipelines; Decision Tree and Gaussian Naive Bayes use the original features. All metrics were evaluated on the same held-out test set. The positive class for precision, recall, F1 and AUC is **benign (1)**.

## 3. GitHub Repository Link

**Repository:** [INSERT GITHUB REPOSITORY URL AFTER UPLOADING]

## 4. Models and Evaluation Metrics

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| SVM | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

These are results observed in the assignment notebook and verified in the BITS Virtual Lab Streamlit dashboard. AUC is calculated from predicted probabilities.

## 5. Observations About Model Performance

| ML Model | Observation |
|---|---|
| Logistic Regression | Highest AUC (0.9954), tied highest accuracy (0.9825), and 112/114 correct test predictions. |
| Decision Tree | Lowest accuracy (0.9211) and MCC (0.8341) of the five models on this split. |
| KNN | Accuracy 0.9561 and recall 0.9722; distance-based method benefits from scaling. |
| Gaussian Naive Bayes | Accuracy 0.9386, but strong AUC of 0.9878 on the held-out test set. |
| SVM | Tied Logistic Regression on accuracy, precision, recall, F1 and MCC; AUC 0.9950. |
| Overall winner | Logistic Regression by a very small AUC margin over SVM; both tie on five other reported metrics. |

**Personal reflection (add your own):** [WRITE 2–3 SENTENCES EXPLAINING WHAT YOU LEARNED AND ANY LIMITATIONS YOU NOTICED.]

These results depend on the particular split and model settings; they do not establish universal superiority. Because the positive class is benign, recall here is **not** malignant-case sensitivity.

## 6. Streamlit Application

The application supports CSV upload, model selection, six evaluation metrics, a confusion matrix, and prediction results. The uploaded test CSV must include all 30 input columns and a `target` column to compute evaluation metrics.

Run from the project folder:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Upload `test_data.csv` and select each classifier. Saved `.joblib` files should only be loaded from trusted sources and should be used with compatible scikit-learn versions.

## 7. Repository Structure

```text
ML-Assignment/
├── ML_Assignment.ipynb
├── app_creation.ipynb
├── app.py
├── requirements.txt
├── README.md
├── model_comparison.csv
├── test_data.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## 8. Deployment

**Streamlit Community Cloud URL:** [INSERT LIVE APP URL AFTER DEPLOYMENT]

Deploy `app.py` from the GitHub repository on Streamlit Community Cloud. Verify that the CSV upload and all five models work after deployment.

## 9. Conclusion

Five classification models were implemented and compared on the Wisconsin Diagnostic Breast Cancer dataset. Logistic Regression and SVM achieved the highest test accuracy (0.9825), and Logistic Regression had the highest AUC (0.9954). The Streamlit dashboard demonstrates model selection, evaluation, and prediction on held-out test data.

*Educational demonstration only; not a clinical diagnostic tool.*
