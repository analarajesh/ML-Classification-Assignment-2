
# Machine Learning Classification Assignment 2

**BITS Pilani | M.Tech DSE | Machine Learning**

## 1. Problem Statement

In this assignment, I worked with a classification dataset and compared five machine learning algorithms. The main aim was to train all five models on the same data, check how well they performed on test data, and display the results through a Streamlit application.

The models used were Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM).

The Streamlit app allows a user to upload the test dataset, choose a model, and see its evaluation metrics and confusion matrix.

## 2. Dataset Description

I used the **Wisconsin Diagnostic Breast Cancer** dataset, available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) and through scikit-learn.

- Total samples: **569**
- Input features: **30** numerical measurements
- Target classes: **Malignant (0)** and **Benign (1)**
- Malignant samples: **212**
- Benign samples: **357**
- Missing values: **None**

The features describe measurements of cell nuclei, such as radius, texture, perimeter, and area. I used an 80:20 train-test split with `random_state=42`, giving **455 training samples** and **114 test samples**. The test set was kept separate for evaluation.

For Logistic Regression, KNN, and SVM, I used `StandardScaler` inside a pipeline because these models can be affected by feature scales. The Decision Tree and Gaussian Naive Bayes models were trained without that scaling pipeline.

## 3. Project Links

**GitHub repository:** https://github.com/analarajesh/ML-Classification-Assignment-2

**Live Streamlit app:** https://ml-classification-assignment-2-4osmnv7r3juxgsl4wwhdvn.streamlit.app/

## 4. Models Used and Evaluation Results

I evaluated all five models using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). The metrics below are from the same test set of 114 samples. In the code, class 1 (Benign) is the positive class for Precision, Recall, F1, and AUC.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| K-Nearest Neighbors | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

## 5. Observations About Model Performance

| Model | My observation |
|---|---|
| **Logistic Regression** | This model performed very well, with 98.25% accuracy. It made only two incorrect predictions out of 114 test samples. Its AUC was also the highest among the five models. |
| **Decision Tree** | The Decision Tree was easy to understand as a model, but it gave the lowest accuracy (92.11%) and MCC (0.8341) in this comparison. It made more mistakes than the other models. |
| **KNN** | KNN achieved 95.61% accuracy. It performed better than Decision Tree and Naive Bayes on accuracy, although it did not match Logistic Regression or SVM. |
| **Gaussian Naive Bayes** | Naive Bayes gave 93.86% accuracy. Its AUC was still quite high (0.9878), even though its accuracy was lower than the top models. |
| **SVM** | SVM matched Logistic Regression on accuracy, precision, recall, F1, and MCC. Its AUC was 0.9950, just below Logistic Regression. |

### Overall winner

**Logistic Regression** was my choice for the best model on this test split. Logistic Regression and SVM both reached **98.25% accuracy**, but Logistic Regression had a slightly higher AUC (**0.9954** compared with **0.9950**). Logistic Regression is also relatively straightforward to explain.

These are results from one train-test split, so they do not prove that the same model will always be best on new data.

## 6. Streamlit Application

I built the dashboard using Streamlit. It has the following features:

- Upload the test dataset as a CSV file.
- Select one of the five trained models from a dropdown.
- View the uploaded data.
- See all six evaluation metrics.
- View the confusion matrix and prediction results.

The app loads the trained models saved in the `model/` folder. The file `test_data.csv` contains the test features and their target labels so that the app can calculate evaluation metrics.

### Running the app

From the project folder, install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, upload `test_data.csv`, and select a model to see its results.

## 7. Project Files

```text
ML-Classification-Assignment-2/
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

## 8. Conclusion

This assignment helped me compare different classification algorithms using the same dataset and evaluation measures. Logistic Regression and SVM gave the best accuracy, while Decision Tree had the lowest accuracy in my experiments. I also learned how to save trained models and use them in a Streamlit dashboard instead of only displaying results in a notebook.

The app was tested in the BITS Virtual Lab and deployed through Streamlit Community Cloud.

*This project is for learning and demonstration purposes, not for medical diagnosis.*
