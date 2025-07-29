# Music Genre Classification

This project aims to classify music genres using machine learning techniques. The process involves extracting audio features, preprocessing the data, training various classification models, and evaluating their performance.

## Dataset

The project utilizes a dataset containing audio features (likely MFCCs) and corresponding genre labels. The `data.csv` file is used as the primary data source.

## Methods

### 1. Feature Extraction (MFCCs)

While the exact feature extraction code is not present in `MFCC.ipynb`, the notebook's name suggests that Mel-frequency Cepstral Coefficients (MFCCs) are the primary audio features used. MFCCs are widely used in audio processing to represent the spectral characteristics of sound.

### 2. Data Preprocessing

- **Loading Data**: The `data.csv` file is loaded into a pandas DataFrame.
- **Label Encoding**: The categorical genre labels are converted into numerical representations using `LabelEncoder` from `sklearn.preprocessing`. The mapping of original labels to encoded integers is saved in `label_classes.json`.
- **Feature and Target Separation**: The dataset is split into features (X) and target labels (Y).
- **Train-Test Split**: The data is divided into training and testing sets using `train_test_split` with an 80/20 ratio and a `random_state` for reproducibility.
- **Feature Scaling**: `StandardScaler` is applied to the features to normalize them, which is crucial for many machine learning algorithms.

### 3. Model Training and Evaluation

The project explores several classification models and uses `GridSearchCV` for hyperparameter tuning to find the best parameters for each model based on accuracy.

**Models Tuned:**
- **Transfer Learning `Xception` Model**
    - *Trained of spectrogram images of Music*
    - *Best Score of 68% Validation Accuracy (Low due to Extremely small dataset 1000 images)*

- **Random Forest Classifier**: Tuned parameters include `n_estimators` (number of trees) and `max_depth`.
  - *Best Parameters Example*: `{'max_depth': 20, 'n_estimators': 200}`
  - *Best CV Score Example*: `0.712`

- **Support Vector Machine (SVM)**: Tuned parameters include `C` (regularization parameter) and `kernel`.
  - *Best Parameters Example*: `{'C': 10, 'gamma': 'scale', 'kernel': 'rbf}`
  - *Best CV Score Example*: `0.731`

- **K-Nearest Neighbors (KNN)**: Tuned parameters include `n_neighbors` and `weights`.
  - *Best Parameters Example*: `{'n_neighbors': 3, 'weights': 'distance}`
  - *Best CV Score Example*: `0.651`

- **Multi-layer Perceptron (MLP) Classifier**: Tuned parameters include `hidden_layer_sizes`, `alpha` (L2 regularization), and `activation` function.
  - *Best Parameters Example*: `{'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (128, 64)}`
  - *Best CV Score Example*: `0.720`

### 4. Ensemble Learning (Voting Classifier)

A `VotingClassifier` with 'soft' voting is used to combine the predictions of the top-performing models: Random Forest, SVM, and MLP. This ensemble approach leverages the strengths of individual models to potentially achieve higher overall accuracy.

## Results

The `SVM` achieved an accuracy of approximately **0.92** on the test set. A detailed classification report provides precision, recall, and F1-score for each genre class, along with macro and weighted averages.

**Example Classification Report:**

```
    precision    recall  f1-score   support

           0       0.91      0.97      0.94       207
           1       0.96      0.94      0.95       203
           2       0.90      0.87      0.89       221
           3       0.90      0.90      0.90       193
           4       0.97      0.89      0.93       190
           5       0.92      0.96      0.94       217
           6       0.95      0.93      0.94       216
           7       0.92      0.92      0.92       199
           8       0.88      0.94      0.91       178
           9       0.84      0.83      0.83       174

    accuracy                           0.92      1998
   macro avg       0.92      0.92      0.92      1998
weighted avg       0.92      0.92      0.92      1998
```

This report indicates varying performance across different genres, with some classes showing higher precision/recall than others. The overall accuracy and F1-scores suggest a reasonably good performance for the music genre classification task.