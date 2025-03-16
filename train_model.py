import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_and_save_model():
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Create a simplified dataset with our 7 features
    # We'll select relevant features and simulate others
    feature_indices = {
        'mean radius': 0,  # Proxy for tumor size
        'mean texture': 1,   # Proxy for skin changes
        'mean perimeter': 2   # Related to tumor characteristics
    }
    
    # Create a simplified dataset
    df = pd.DataFrame(X, columns=data.feature_names)
    simplified_data = pd.DataFrame()
    
    # Simulated mapping to our features
    simplified_data['Age'] = np.random.uniform(18, 100, len(y))  # Simulated age
    simplified_data['Family History'] = np.random.choice([0, 1, 0.5], len(y))  # 0: No, 1: Yes, 0.5: Unknown
    simplified_data['Tumor Size'] = df['mean radius'] * 2  # Approximate mm conversion
    simplified_data['Lymph Node'] = np.random.choice([0, 1, 0.5], len(y))  # Simulated
    simplified_data['Menopausal'] = np.random.choice([0, 1, 0.5], len(y))  # Simulated
    simplified_data['Breast Pain'] = np.random.choice([0, 1], len(y))  # Simulated
    simplified_data['Skin Changes'] = (df['mean texture'] > df['mean texture'].mean()).astype(float)
    
    X_simplified = simplified_data.values
    y = y  # 0: malignant, 1: benign (note: we'll invert prediction later to match app)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_simplified, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # # Save the model and scaler
    joblib.dump(model, 'patient_self_test_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    train_and_save_model()