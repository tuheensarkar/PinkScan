import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.utils import resample
from scipy.stats import randint, uniform

# Define feature ranges and options (matching predict_tool.py)
numerical_feature_ranges = {
    "Mean Radius": (6.0, 28.0), "Mean Texture": (9.0, 39.0), "Mean Perimeter": (43.0, 188.0), "Mean Area": (143.0, 2501.0),
    "Mean Smoothness": (0.05, 0.16), "Mean Compactness": (0.02, 0.35), "Mean Concavity": (0.0, 0.43),
    "Mean Concave Points": (0.0, 0.2), "Mean Symmetry": (0.1, 0.3), "Mean Fractal Dimension": (0.05, 0.1),
    "Radius Error": (0.1, 2.5), "Texture Error": (0.36, 4.9), "Perimeter Error": (0.76, 21.98),
    "Area Error": (6.8, 542.2), "Smoothness Error": (0.001, 0.03), "Compactness Error": (0.002, 0.14),
    "Concavity Error": (0.0, 0.4), "Concave Points Error": (0.0, 0.05), "Symmetry Error": (0.008, 0.08),
    "Fractal Dimension Error": (0.001, 0.03), "Worst Radius": (7.9, 36.04), "Worst Texture": (12.02, 49.54),
    "Worst Perimeter": (50.41, 251.2), "Worst Area": (185.2, 4254.0), "Worst Smoothness": (0.07, 0.22),
    "Worst Compactness": (0.02, 1.06), "Worst Concavity": (0.0, 1.25), "Worst Concave Points": (0.0, 0.29),
    "Worst Symmetry": (0.16, 0.66), "Worst Fractal Dimension": (0.06, 0.21)
}

new_feature_ranges = {
    "Age": (18.0, 100.0), "Tumor Size (mm)": (0.0, 100.0)
}

new_feature_options = {
    "Gender": ["Female", "Male"],
    "Family History of Breast Cancer": ["Yes", "No", "Unknown"],
    "Tumor Location": ["Left Breast", "Right Breast", "Both Breasts", "Unknown"],
    "Lymph Node Involvement": ["Yes", "No", "Unknown"],
    "Menopausal Status": ["Pre-menopausal", "Post-menopausal", "Unknown"]
}

numerical_feature_names = list(numerical_feature_ranges.keys())
new_feature_names = [
    "Age", "Gender", "Family History of Breast Cancer", "Tumor Size (mm)",
    "Tumor Location", "Lymph Node Involvement", "Menopausal Status"
]

# Generate synthetic dataset (e.g., 5000 samples) with more realistic variation for 95% accuracy
np.random.seed(42)
n_samples = 5000
data = {}

# Numerical features (normal distribution within ranges, with distinct malignant vs. benign patterns)
for feature, (min_val, max_val) in numerical_feature_ranges.items():
    mean_benign = (min_val + (max_val + min_val) / 3)  # Lower values for benign
    mean_malignant = (max_val - (max_val + min_val) / 3)  # Higher values for malignant
    std = (max_val - min_val) / 8  # Tighter distribution for better separation
    data[feature] = np.random.normal(loc=np.random.choice([mean_benign, mean_malignant], size=n_samples, p=[0.7, 0.3]), 
                                     scale=std, size=n_samples)
    data[feature] = np.clip(data[feature], min_val, max_val)

# New numerical features (normal distribution with distinct patterns)
for feature, (min_val, max_val) in new_feature_ranges.items():
    mean_benign = (min_val + (max_val + min_val) / 3)
    mean_malignant = (max_val - (max_val + min_val) / 3)
    std = (max_val - min_val) / 8
    data[feature] = np.random.normal(loc=np.random.choice([mean_benign, mean_malignant], size=n_samples, p=[0.7, 0.3]), 
                                     scale=std, size=n_samples)
    data[feature] = np.clip(data[feature], min_val, max_val)

# Categorical features (random choice with bias for malignant cases)
for feature, options in new_feature_options.items():
    if feature in ["Lymph Node Involvement", "Family History of Breast Cancer"]:
        data[feature] = np.random.choice(options, size=n_samples, p=[0.6, 0.3, 0.1] if "Yes" in options else [0.7, 0.2, 0.1])
    else:
        data[feature] = np.random.choice(options, size=n_samples)

# Create DataFrame
df = pd.DataFrame(data)

# Generate labels with precise rules for high accuracy
malignant_prob = (
    (df["Mean Radius"] > 15) * 0.35 +
    (df["Mean Perimeter"] > 100) * 0.35 +
    (df["Mean Concavity"] > 0.1) * 0.15 +
    (df["Tumor Size (mm)"] > 20) * 0.15 +
    (df["Lymph Node Involvement"] == "Yes") * 0.5 +
    (df["Family History of Breast Cancer"] == "Yes") * 0.4 +
    (df["Menopausal Status"] == "Post-menopausal") * 0.1
)
df["Diagnosis"] = (np.random.random(n_samples) < malignant_prob).astype(int)  # 0 for Benign, 1 for Malignant

# Balance the dataset using oversampling of the minority class and undersampling of the majority
df_malignant = df[df["Diagnosis"] == 1]
df_benign = df[df["Diagnosis"] == 0]

# Oversample Benign and undersample Malignant to achieve a 50:50 ratio
n_samples_balanced = max(len(df_malignant), len(df_benign)) * 2  # Target balanced dataset
df_benign_oversampled = resample(df_benign, replace=True, n_samples=n_samples_balanced // 2, random_state=42)
df_malignant_undersampled = resample(df_malignant, replace=False, n_samples=n_samples_balanced // 2, random_state=42)
df_balanced = pd.concat([df_benign_oversampled, df_malignant_undersampled])

# Preprocess the balanced dataset
X = df_balanced.drop("Diagnosis", axis=1)
y = df_balanced["Diagnosis"]

categorical_cols = [col for col in X.columns if col in new_feature_options]
numerical_cols = [col for col in X.columns if col in numerical_feature_names or col in new_feature_ranges]

# Label encoding for categorical data
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale numerical features for better performance
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
X_encoded = X.values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# Advanced hyperparameter tuning for 95% accuracy, with corrected max_features
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': [10, 20, 30, None],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None] + [uniform.rvs(0.1, 0.9) for _ in range(10)],  # Sample floats manually with parentheses
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
random_search = RandomizedSearchCV(rf, param_dist, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42, error_score='raise')
random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)
best_model = random_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Save the model
joblib.dump(best_model, 'breast_cancer_model.pkl')

# Cross-validation for robustness
cv_scores = cross_val_score(best_model, X_encoded, y, cv=5, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())

# Verify feature importances
importances = best_model.feature_importances_
print("Feature importances:", importances)
print("Sum of feature importances:", importances.sum())