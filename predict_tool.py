import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os
from deep_translator import GoogleTranslator
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import ast
import json
from datetime import datetime, timedelta

# Ensure set_page_config is the very first Streamlit command
st.set_page_config(page_title="PinkScan - Predict Tool", page_icon=":hospital:", layout="wide")

# Cache model loading
@st.cache_resource
def load_model():
    return joblib.load('breast_cancer_model.pkl')  # Model trained on 30 numerical + 7 new features (37 total)

# Load model
model = load_model()

# Feature names and ranges (existing numerical features from Wisconsin Dataset)
numerical_feature_names = [
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
    "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    "Radius Error", "Texture Error", "Perimeter Error", "Area Error", "Smoothness Error",
    "Compactness Error", "Concavity Error", "Concave Points Error", "Symmetry Error", "Fractal Dimension Error",
    "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness",
    "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
]

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

# Selected new features (7 total, fast to obtain, impactful)
new_feature_names = [
    "Age", "Gender", "Family History of Breast Cancer", "Tumor Size (mm)", "Tumor Location",
    "Lymph Node Involvement", "Menopausal Status"
]

new_feature_options = {
    "Gender": ["Female", "Male"],
    "Family History of Breast Cancer": ["Yes", "No", "Unknown"],
    "Tumor Location": ["Left Breast", "Right Breast", "Both Breasts", "Unknown"],
    "Lymph Node Involvement": ["Yes", "No", "Unknown"],
    "Menopausal Status": ["Pre-menopausal", "Post-menopausal", "Unknown"]
}

new_feature_ranges = {
    "Age": (18.0, 100.0), "Tumor Size (mm)": (0.0, 100.0)
}

LANGUAGES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'bn': 'Bengali', 'hi': 'Hindi'
}

# Apply light theme CSS after set_page_config
st.markdown("""
<style>
.stApp {
    background-color: #fff;
    color: #000000;
}
.stButton>button {
    background-color: #ff69b4;
    color: #fff;
    border-radius: 25px;
    padding: 10px 20px;
    border: none;
    font-weight: bold;
    font-size: 1rem;
}
.stButton>button:hover {
    background-color: #ff4d9e;
    color: #fff;
}
.stHeader {
    color: #ff69b4;
    font-size: 2.5rem;
    text-align: center;
    font-weight: bold;
}
.stSubheader {
    color: #000000;
    font-size: 1.5rem;
}
.stText {
    color: #808080;
}
.stWarning {
    background-color: #ffebee;
    border: 1px solid #ffcdd2;
    border-radius: 5px;
    padding: 10px;
    color: #000000;
}
.stSuccess {
    background-color: #e8f5e9;
    border: 1px solid #c8e6c9;
    border-radius: 5px;
    padding: 10px;
    color: #000000;
}
.stError {
    background-color: #ffebee;
    border: 1px solid #ffcdd2;
    border-radius: 5px;
    padding: 10px;
    color: #000000;
}
.chatbot-container {
    border: 1px solid #ff69b4;
    border-radius: 10px;
    padding: 15px;
    margin-top: 20px;
    background-color: #fff;
}
.chatbot-message {
    margin: 10px 0;
    padding: 10px;
    border-radius: 5px;
    background-color: #f8f9fa;
    color: #000000;
}
.chatbot-user {
    background-color: #e9ecef;
    text-align: right;
    color: #000000;
}
.chatbot-bot {
    background-color: #fff;
    text-align: left;
    color: #000000;
}
.chatbot-message strong {
    color: #ff69b4;
    font-weight: bold;
    margin-right: 5px;
}
.sidebar .sidebar-content {
    background-color: #f8f9fa;
    color: #000000;
}
.sidebar .sidebar-content h2, .sidebar .sidebar-content h3 {
    color: #ff69b4;
}
.sidebar .sidebar-content p, .sidebar .sidebar-content li {
    color: #000000;
}
</style>
""", unsafe_allow_html=True)

# Translation functions
@st.cache_data
def translate_text(text, lang):
    translator = GoogleTranslator(source='auto', target=lang)
    try:
        if isinstance(text, list):
            return [translator.translate(item) for item in text if isinstance(item, str) and len(item) <= 5000]
        return translator.translate(text) if isinstance(text, str) and len(text) <= 5000 else text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text if isinstance(text, str) else text

def translate(_text):
    language = st.session_state.get('language', 'English')
    return translate_text(_text, list(LANGUAGES.keys())[list(LANGUAGES.values()).index(language)])

def encode_features(numerical_features, new_features):
    # Ensure numerical_features is a list or 1D array with exactly 30 values
    if len(numerical_features) != 30:
        raise ValueError("Numerical features must contain exactly 30 values.")
    
    # Ensure numerical_features is a 1D list or array (flatten if needed)
    if isinstance(numerical_features, np.ndarray):
        if numerical_features.ndim > 1:
            numerical_features = numerical_features.flatten()
        numerical_features = numerical_features.tolist()
    elif isinstance(numerical_features, list):
        if any(isinstance(x, (list, np.ndarray)) for x in numerical_features):
            numerical_features = [float(x[0]) if isinstance(x, (list, np.ndarray)) else float(x) for x in numerical_features]
    else:
        raise ValueError("Numerical features must be a list or NumPy array.")
    
    # Convert to 1D NumPy array for consistency
    numerical_features = np.array(numerical_features, dtype=float).flatten()
    if numerical_features.ndim != 1 or len(numerical_features) != 30:
        st.error(f"Numerical features shape mismatch: expected (30,), got {numerical_features.shape}")
        raise ValueError("Numerical features must be a 1D array with 30 values.")
    
    all_features = numerical_features.copy()
    feature_mapping = {
        "Gender": {"Female": 0.0, "Male": 1.0},
        "Family History of Breast Cancer": {"No": 0.0, "Yes": 1.0, "Unknown": 2.0},
        "Tumor Location": {"Left Breast": 0.0, "Right Breast": 1.0, "Both Breasts": 2.0, "Unknown": 3.0},
        "Lymph Node Involvement": {"No": 0.0, "Yes": 1.0, "Unknown": 2.0},
        "Menopausal Status": {"Pre-menopausal": 0.0, "Post-menopausal": 1.0, "Unknown": 2.0}
    }
    
    for i, feature in enumerate(new_feature_names):
        if feature in feature_mapping:
            all_features = np.append(all_features, feature_mapping[feature][new_features[i]])
        else:
            all_features = np.append(all_features, float(new_features[i]) if isinstance(new_features[i], (int, float)) else 0.0)
    
    # Ensure all_features is a 1D array
    all_features = all_features.flatten()
    if all_features.ndim != 1 or len(all_features) != 37:
        st.error(f"Encoded features shape mismatch: expected (37,), got {all_features.shape}")
        raise ValueError("Encoded features must be a 1D array with 37 values.")
    
    return all_features

@st.cache_data
def predict_breast_cancer(encoded_features):
    # Ensure encoded_features is a 1D array and reshape to (1, 37) for prediction
    if not isinstance(encoded_features, np.ndarray) or encoded_features.ndim != 1 or len(encoded_features) != 37:
        st.error(f"Encoded features shape mismatch: expected (37,), got {encoded_features.shape if isinstance(encoded_features, np.ndarray) else 'invalid'}")
        raise ValueError("Encoded features must be a 1D array with 37 values.")
    
    encoded_features = encoded_features.reshape(1, -1)  # Reshape to (1, 37)
    prediction = model.predict(encoded_features)
    probability = model.predict_proba(encoded_features)[0]
    return "Malignant" if prediction[0] == 1 else "Benign", probability[1] * 100  # Updated for class 1 = Malignant

@st.cache_data
def generate_pdf_report(patient_id, result, numerical_features, new_features, risk_score):
    pdf = FPDF()
    pdf.add_page()
    
    # Set consistent margins (20mm on all sides)
    margin = 20
    page_width = 210 - (2 * margin)  # Usable width: 170mm
    page_height = 297 - (2 * margin)  # Usable height: 257mm
    
    # Page layout with pink border
    def apply_page_layout():
        pdf.set_margins(left=margin, top=margin, right=margin)
        pdf.set_line_width(0.5)
        pdf.set_draw_color(255, 105, 180)  # Pink (#ff69b4)
        pdf.rect(margin, margin, page_width, page_height)
    
    apply_page_layout()
    
    # Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 105, 180)  # Pink
    pdf.cell(0, 10, "PinkScan - Breast Cancer Prediction Report", 0, 1, "C")
    pdf.ln(5)
    
    # Patient Information
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)  # Black
    pdf.cell(0, 8, "Patient Information", 0, 1)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, f"Patient ID: {patient_id}", 0, 1)
    pdf.cell(0, 8, f"Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(5)
    
    # Prediction Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Prediction Summary", 0, 1)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, f"Current Diagnosis: {result}", 0, 1)
    pdf.cell(0, 8, f"Current Risk Score: {risk_score:.2f}% likelihood of malignant tumor", 0, 1)
    pdf.ln(10)
    
    # Bar Chart (Initial vs Current Comparison) - First Page
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Comparison: Initial vs Current Feature Values", 0, 1, "C")
    pdf.ln(5)
    
    # Generate comparative bar chart
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import os
    import tempfile
    
    history = get_patient_history(patient_id)
    if not history.empty:
        initial_entry = history.iloc[0]
        try:
            initial_numerical = json.loads(initial_entry['numerical_features'])
        except:
            initial_numerical = numerical_features  # Fallback if parsing fails
    else:
        initial_numerical = numerical_features  # No history, use current as initial
    
    current_numerical = numerical_features
    all_feature_names = numerical_feature_names + new_feature_names
    initial_features = initial_numerical + [float(x) if isinstance(x, (int, float)) else 0.0 for x in new_features]
    current_features = current_numerical + [float(x) if isinstance(x, (int, float)) else 0.0 for x in new_features]
    
    # Create DataFrame for plotting
    data = pd.DataFrame({
        'Feature': all_feature_names,
        'Initial Value': initial_features,
        'Current Value': current_features
    })
    
    # Plot bar chart
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(data['Feature']))
    plt.bar(x - bar_width/2, data['Initial Value'], bar_width, label='Initial', color='#FF6347')
    plt.bar(x + bar_width/2, data['Current Value'], bar_width, label='Current', color='#4682B4')
    plt.xlabel('Features', fontsize=10)
    plt.ylabel('Values', fontsize=10)
    plt.title('Initial vs Current Feature Values', fontsize=12, color='#ff69b4')
    plt.xticks(x, data['Feature'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save bar chart to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        plt.savefig(temp_file.name, format='png', bbox_inches='tight', dpi=150)
        temp_file_path = temp_file.name
    plt.close()
    
    # Add bar chart to PDF
    pdf.image(temp_file_path, x=margin, y=pdf.get_y(), w=page_width, h=120)  # Fit within pink border
    pdf.ln(130)  # Ensure enough space after bar chart
    
    # Clean up the temporary file
    os.unlink(temp_file_path)
    
    # Pie Chart and Bar Chart - Second Page (Up-Down Layout)
    pdf.add_page()
    apply_page_layout()
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Prediction and Feature Analysis", 0, 1, "C")
    pdf.ln(5)
    
    # Load pie chart and bar chart images
    pie_chart_path = f"pie_chart_{patient_id}.png"
    bar_chart_path = f"bar_chart_{patient_id}.png"
    
    # Stack pie chart and bar chart vertically
    if os.path.exists(pie_chart_path):
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, "Risk Breakdown", 0, 1, "C")
        pdf.image(pie_chart_path, x=margin, y=pdf.get_y(), w=page_width, h=80)  # Fit within pink border
        pdf.ln(90)
    
    if os.path.exists(bar_chart_path):
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, "Top 5 Most Important Features", 0, 1, "C")
        pdf.image(bar_chart_path, x=margin, y=pdf.get_y(), w=page_width, h=80)  # Fit within pink border
        pdf.ln(90)
    
    # Clinical Features Table
    pdf.add_page()
    apply_page_layout()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Clinical Features", 0, 1)
    pdf.set_font("Helvetica", size=10)
    col_widths = [page_width * 0.5, page_width * 0.5]
    headers = ["Feature", "Value"]
    pdf.set_fill_color(220, 220, 220)
    pdf.set_draw_color(128, 128, 128)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, 1, 0, "C", 1)
    pdf.ln(8)
    
    pdf.set_fill_color(255, 255, 255)
    fill = False
    for name, value in zip(new_feature_names, new_features):
        value_str = str(value) if isinstance(value, str) else f"{float(value):.2f}"
        pdf.cell(col_widths[0], 8, name, 1, 0, "L", fill)
        pdf.cell(col_widths[1], 8, value_str, 1, 0, "L", fill)
        pdf.ln(8)
        fill = not fill
        if pdf.get_y() > page_height - margin - 20:
            pdf.add_page()
            apply_page_layout()
            pdf.set_font("Helvetica", size=10)
    
    # Numerical Features Table
    pdf.add_page()
    apply_page_layout()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Numerical Features", 0, 1)
    pdf.set_font("Helvetica", size=10)
    col_widths = [page_width * 0.5, page_width * 0.5]
    headers = ["Feature", "Value"]
    pdf.set_fill_color(220, 220, 220)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, 1, 0, "C", 1)
    pdf.ln(8)
    
    pdf.set_fill_color(255, 255, 255)
    fill = False
    for name, value in zip(numerical_feature_names, numerical_features):
        value_str = f"{float(value):.2f}"
        pdf.cell(col_widths[0], 8, name, 1, 0, "L", fill)
        pdf.cell(col_widths[1], 8, value_str, 1, 0, "L", fill)
        pdf.ln(8)
        fill = not fill
        if pdf.get_y() > page_height - margin - 20:
            pdf.add_page()
            apply_page_layout()
            pdf.set_font("Helvetica", size=10)
    
    # Historical Comparison and Report (with Recommendations)
    history = get_patient_history(patient_id)
    if not history.empty:
        pdf.add_page()
        apply_page_layout()
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Historical Comparison Report", 0, 1)
        pdf.ln(5)
        
        initial_entry = history.iloc[0]
        current_entry = history.iloc[-1]
        
        # Summary
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, "Summary", 0, 1)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 8, f"Initial Diagnosis: {initial_entry['prediction']} (Date: {initial_entry['date']})", 0, 1)
        pdf.cell(0, 8, f"Initial Risk Score: {initial_entry['risk_score']:.2f}%", 0, 1)
        pdf.cell(0, 8, f"Current Diagnosis: {current_entry['prediction']} (Date: {current_entry['date']})", 0, 1)
        pdf.cell(0, 8, f"Current Risk Score: {current_entry['risk_score']:.2f}%", 0, 1)
        pdf.ln(5)
        
        # Feature Comparison Table
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, "Feature Comparison (Initial vs. Current)", 0, 1)
        pdf.set_font("Helvetica", size=9)
        col_widths = [page_width * 0.4, page_width * 0.3, page_width * 0.3]
        headers = ["Feature", "Initial Value", "Current Value"]
        pdf.set_fill_color(220, 220, 220)
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 8, header, 1, 0, "C", 1)
        pdf.ln(8)
        
        pdf.set_fill_color(255, 255, 255)
        fill = False
        try:
            initial_numerical = json.loads(initial_entry['numerical_features'])
            initial_new = json.loads(initial_entry['new_features'])
        except:
            initial_numerical = numerical_features
            initial_new = new_features
        
        all_feature_names = numerical_feature_names + new_feature_names
        initial_features = initial_numerical + initial_new
        current_features = numerical_features + new_features
        
        for name, initial_val, current_val in zip(all_feature_names, initial_features, current_features):
            initial_str = str(initial_val) if isinstance(initial_val, str) else f"{float(initial_val):.2f}"
            current_str = str(current_val) if isinstance(current_val, str) else f"{float(current_val):.2f}"
            pdf.cell(col_widths[0], 8, name, 1, 0, "L", fill)
            pdf.cell(col_widths[1], 8, initial_str, 1, 0, "C", fill)
            pdf.cell(col_widths[2], 8, current_str, 1, 0, "C", fill)
            pdf.ln(8)
            fill = not fill
            if pdf.get_y() > page_height - margin - 20:
                pdf.add_page()
                apply_page_layout()
                pdf.set_font("Helvetica", size=9)
        
        # Analysis
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, "Analysis", 0, 1)
        pdf.set_font("Helvetica", size=10)
        initial_risk = initial_entry['risk_score']
        current_risk = current_entry['risk_score']
        days_elapsed = (datetime.strptime(current_entry['date'], '%Y-%m-%d %H:%M:%S') - 
                        datetime.strptime(initial_entry['date'], '%Y-%m-%d %H:%M:%S')).days
        
        if initial_entry['prediction'] == "Malignant" and current_entry['prediction'] == "Benign":
            pdf.cell(0, 8, "Status: Improving", 0, 1)
            pdf.cell(0, 8, f"Progress: Tumor shifted from malignant to benign over {days_elapsed} days.", 0, 1)
            estimated_days = max(0, 365 - days_elapsed)
            pdf.cell(0, 8, f"Estimated Time to Cure: ~{estimated_days} days with continued treatment.", 0, 1)
        elif initial_entry['prediction'] == "Benign" and current_entry['prediction'] == "Malignant":
            pdf.cell(0, 8, "Status: Worsening", 0, 1)
            pdf.cell(0, 8, f"Progress: Tumor shifted from benign to malignant over {days_elapsed} days.", 0, 1)
            pdf.cell(0, 8, "Action: Immediate medical intervention required.", 0, 1)
        elif initial_risk > current_risk:
            pdf.cell(0, 8, "Status: Improving", 0, 1)
            pdf.cell(0, 8, f"Progress: Risk reduced by {initial_risk - current_risk:.2f}% over {days_elapsed} days.", 0, 1)
            estimated_days = int((90 * (initial_risk - current_risk) / 100))
            pdf.cell(0, 8, f"Estimated Time to Cure: ~{estimated_days} days with continued improvement.", 0, 1)
        elif initial_risk < current_risk:
            pdf.cell(0, 8, "Status: Worsening", 0, 1)
            pdf.cell(0, 8, f"Progress: Risk increased by {current_risk - initial_risk:.2f}% over {days_elapsed} days.", 0, 1)
            pdf.cell(0, 8, "Action: Consult a specialist immediately.", 0, 1)
        else:
            pdf.cell(0, 8, "Status: Stable", 0, 1)
            pdf.cell(0, 8, "Progress: No significant change detected.", 0, 1)
        
        # Recommendations (under Analysis on the same page)
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, "Recommendations", 0, 1)
        pdf.set_font("Helvetica", size=10)
        tips = [
            "Schedule regular mammograms as recommended by your doctor",
            "Perform monthly breast self-examinations",
            "Maintain a healthy lifestyle",
            "Consult an oncologist for personalized advice"
        ]
        for tip in tips:
            pdf.cell(0, 6, f"- {tip}", 0, 1)
            if pdf.get_y() > page_height - margin - 20:
                pdf.add_page()
                apply_page_layout()
                pdf.set_font("Helvetica", size=10)
    
    # Footer
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 10, f"PinkScan v1.0 | Generated on {pd.Timestamp.now().strftime('%Y-%m-%d')}", 0, 0, "C")
    
    pdf_file = f"report_{patient_id}.pdf"
    pdf.output(pdf_file)
    return pdf_file

@st.cache_data
def save_patient_history(patient_id, result, risk_score, numerical_features, new_features):
    history_file = 'patient_history.csv'
    if os.path.exists(history_file):
        try:
            history_df = pd.read_csv(history_file, encoding='utf-8')
        except UnicodeDecodeError:
            history_df = pd.read_csv(history_file, encoding='latin-1')
    else:
        # Initialize empty DataFrame with explicit dtypes to avoid warnings
        history_df = pd.DataFrame(columns=['patient_id', 'prediction', 'risk_score', 'numerical_features', 'new_features', 'date', 'is_initial'])
        history_df = history_df.astype({
            'patient_id': str,
            'prediction': str,
            'risk_score': float,
            'numerical_features': str,
            'new_features': str,
            'date': str,
            'is_initial': bool
        })

    # Check if this is the first entry for this patient
    patient_history = history_df[history_df['patient_id'] == str(patient_id)]
    is_initial = patient_history.empty

    # Ensure features are saved as properly formatted JSON strings
    numerical_features_str = json.dumps(numerical_features)
    new_features_str = json.dumps(new_features)

    # Ensure new_entry has consistent dtypes and includes features
    new_entry = pd.DataFrame({
        'patient_id': [str(patient_id)],
        'prediction': [str(result)],
        'risk_score': [float(risk_score)],
        'numerical_features': [numerical_features_str],  # Store as JSON string
        'new_features': [new_features_str],  # Store as JSON string
        'date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
        'is_initial': [is_initial]
    })

    # Concatenate, ensuring no empty or all-NA entries cause issues
    if not history_df.empty:
        history_df = pd.concat([history_df, new_entry], ignore_index=True)
    else:
        history_df = new_entry

    # Save with explicit encoding and no index
    history_df.to_csv(history_file, index=False, encoding='utf-8')

@st.cache_data
def get_patient_history(patient_id=None):
    history_file = 'patient_history.csv'
    if os.path.exists(history_file):
        try:
            history_df = pd.read_csv(history_file, encoding='utf-8')
        except UnicodeDecodeError:
            history_df = pd.read_csv(history_file, encoding='latin-1')
        if patient_id:
            # Ensure patient_id is string for comparison
            return history_df[history_df['patient_id'].astype(str) == str(patient_id)].sort_values('date')
        return history_df
    return pd.DataFrame(columns=['patient_id', 'prediction', 'risk_score', 'numerical_features', 'new_features', 'date', 'is_initial'])

@st.cache_data
def send_email(email, patient_id, result, risk_score, pdf_file):
    sender_email = "tuheensarkarofficial@gmail.com"  # Replace with your Gmail address
    app_password = "edwb arob jqis cwjc"  # Replace with your App Password (16-character code)

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = f"PinkScan - Breast Cancer Prediction Report for {patient_id}"
    msg.attach(MIMEText(f"Current Prediction: {result}\nCurrent Risk Score: {risk_score:.2f}% likelihood of malignant tumor"))

    try:
        with open(pdf_file, "rb") as f:
            part = MIMEApplication(f.read(), Name=pdf_file)
            part['Content-Disposition'] = f'attachment; filename="{pdf_file}"'
            msg.attach(part)

        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, app_password)
            server.send_message(msg)
            st.success("Email sent successfully using port 587!")
            server.quit()
        except Exception as e:
            st.error(f"Failed to send email using port 587: {str(e)}")
            try:
                server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
                server.ehlo()
                server.login(sender_email, app_password)
                server.send_message(msg)
                st.success("Email sent successfully using port 465!")
                server.quit()
            except Exception as e2:
                st.error(f"Failed to send email using port 465: {str(e2)}")
                st.error("Please check your Gmail credentials, App Password, and network settings.")
    except Exception as e:
        st.error(f"Error preparing email or attaching file: {str(e)}")

def main():
    # Language selection
    if 'language' not in st.session_state:
        st.session_state.language = 'English'
    language = st.sidebar.selectbox("Select Language", options=list(LANGUAGES.values()), 
                                   index=list(LANGUAGES.values()).index(st.session_state.language))
    st.session_state.language = language

    st.title(translate("PinkScan - Breast Cancer Prediction Tool"))
    st.write(translate(
        "Stay ahead with rapid, early detection and smart decisions. This unique AI-powered tool predicts whether a breast tumor is malignant or benign using numerical and key clinical features, minimizing prediction delays while enhancing accuracy and personalization."
    ))

    st.sidebar.header(translate("About PinkScan"))
    st.sidebar.write(translate(
        "PinkScan uses a Random Forest Classifier trained on the Breast Cancer Wisconsin (Diagnostic) Dataset (30 numerical features) and selected clinical data (7 fast-obtainable features) for rapid, accurate predictions. Malignant: Cancerous tumor. Benign: Non-cancerous tumor. Non-Malignant: No cancer present."
    ))

    with st.sidebar.expander(translate("Learn More About PinkScan's Uniqueness")):
        st.markdown("""
            <h2 style='color: #FF69B4;'>What Makes PinkScan Unique</h2>
            <p style='color: #333333;'>
                PinkScan combines cutting-edge image-based numerical features with a curated set of clinical data, ensuring fast predictions while maintaining high accuracy. It introduces innovative features like <strong>Menopausal Status</strong> and <strong>Image Texture Complexity</strong> to enhance personalization and performance, setting it apart from traditional tools.
            </p>
            <h3 style='color: #FF69B4;'>Key Innovations</h3>
            <ul style='color: #333333;'>
                <li>Real-time prediction using only fast-obtainable data.</li>
                <li>Unique feature engineering (e.g., Image Texture Complexity) for improved accuracy.</li>
                <li>Personalized risk assessment with minimal time delays.</li>
            </ul>
        """, unsafe_allow_html=True)

    with st.sidebar.expander(translate("Learn More About Breast Cancer")):
        st.markdown("""
            <h2 style='color: #FF69B4;'>Understanding Breast Cancer</h2>
            <p style='color: #333333;'>
                Breast cancer is a type of cancer that develops in the cells of the breasts. It is the most common cancer among women worldwide, but it can also affect men.
            </p>
            <h3 style='color: #FF69B4;'>Symptoms</h3>
            <ul style='color: #333333;'>
                <li>A lump or thickening in the breast or underarm.</li>
                <li>Changes in the size, shape, or appearance of the breast.</li>
                <li>Nipple discharge or inversion.</li>
                <li>Redness or pitting of the breast skin (like an orange peel).</li>
            </ul>
            <h3 style='color: #FF69B4;'>Prevention</h3>
            <ul style='color: #333333;'>
                <li>Maintain a healthy weight.</li>
                <li>Exercise regularly.</li>
                <li>Limit alcohol consumption.</li>
                <li>Avoid smoking.</li>
            </ul>
        """, unsafe_allow_html=True)

    patient_id = st.text_input(translate("Enter Patient ID"), placeholder="e.g., PAT12345")

    # Numerical Features (Wisconsin Dataset)
    st.header(translate("Numerical Features (Wisconsin Dataset)"))
    numerical_features = []
    for group_name, features in {
        translate("Mean Features"): numerical_feature_names[:10],
        translate("Error Features"): numerical_feature_names[10:20],
        translate("Worst Features"): numerical_feature_names[20:]
    }.items():
        with st.expander(f"{group_name}"):
            for feature in features:
                min_val, max_val = numerical_feature_ranges[feature]
                value = st.number_input(f"{feature}", min_value=float(min_val), max_value=float(max_val), 
                                       value=float(min_val), help=f"Enter a value between {min_val} and {max_val}")
                if value < min_val or value > max_val:
                    st.warning(f"Value for {feature} is outside typical range ({min_val}-{max_val}).")
                numerical_features.append(float(value))  # Ensure float and 1D

    # Selected New Features (fast to obtain, impactful)
    st.header(translate("Key Clinical Features"))
    new_features = []
    with st.expander(translate("Patient and Clinical Information")):
        age = st.number_input(translate("Age"), min_value=float(new_feature_ranges["Age"][0]), max_value=float(new_feature_ranges["Age"][1]), value=30.0)
        gender = st.selectbox(translate("Gender"), options=new_feature_options["Gender"], index=0)
        family_history = st.selectbox(translate("Family History of Breast Cancer"), options=new_feature_options["Family History of Breast Cancer"], index=0)
        tumor_size = st.number_input(translate("Tumor Size (mm)"), min_value=float(new_feature_ranges["Tumor Size (mm)"][0]), max_value=float(new_feature_ranges["Tumor Size (mm)"][1]), value=10.0)
        tumor_location = st.selectbox(translate("Tumor Location"), options=new_feature_options["Tumor Location"], index=0)
        lymph_nodes = st.selectbox(translate("Lymph Node Involvement"), options=new_feature_options["Lymph Node Involvement"], index=0)
        menopausal_status = st.selectbox(translate("Menopausal Status"), options=new_feature_options["Menopausal Status"], index=0)
        new_features.extend([float(age), gender, family_history, float(tumor_size), tumor_location, lymph_nodes, menopausal_status])

    if st.button(translate("Predict")):
        if not patient_id:
            st.warning(translate("Please enter a Patient ID."))
        elif len(numerical_features) != 30:
            st.error(translate("Please provide all 30 numerical features."))
        else:
            try:
                # Convert numerical_features to 1D NumPy array and ensure shape (30,)
                numerical_features = np.array(numerical_features, dtype=float).flatten()
                if numerical_features.ndim != 1 or len(numerical_features) != 30:
                    st.error(f"Numerical features shape mismatch: expected (30,), got {numerical_features.shape}")
                    raise ValueError("Numerical features must be a 1D array with 30 values.")
                
                # Store all prediction-related data in session state to ensure availability
                st.session_state.prediction_data = {
                    'numerical_features': numerical_features.tolist(),
                    'new_features': new_features,
                    'patient_id': patient_id,
                    'result': None,  # Will be updated after prediction
                    'risk_score': None  # Will be updated after prediction
                }

                # Encode all 37 features (30 numerical + 7 new) for prediction
                encoded_features = encode_features(numerical_features, new_features)
                result, risk_score = predict_breast_cancer(encoded_features)
                
                # Update session state with prediction results
                st.session_state.prediction_data['result'] = result
                st.session_state.prediction_data['risk_score'] = risk_score
                
                # Determine risk level
                risk_level = "Low" if risk_score < 33.33 else "Medium" if risk_score < 66.67 else "High"
                
                if result == "Malignant":
                    st.error(f"Patient ID: {patient_id}\n{translate('Prediction')}: {result}")
                else:
                    st.success(f"Patient ID: {patient_id}\n{translate('Prediction')}: {result}")

                st.write(f"{translate('Risk Score')}: {risk_score:.2f}% ({translate(risk_level)} risk)")
                st.write(f"{translate('Probability of Malignancy')}: {risk_score:.2f}%")

                current_patient = pd.DataFrame({
                    'patient_id': [patient_id],
                    'prediction': [result],
                    'risk_score': [risk_score],
                    'date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
                })
                st.subheader(translate("Current Patient Details"))
                st.write(current_patient)

                st.header(translate("Data Analytics"))
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(translate("Numerical Feature Distribution by Diagnosis"))
                    fig1 = plot_histogram(numerical_features, result)
                    histogram_path = f"histogram_{patient_id}.png"
                    fig1.savefig(histogram_path, bbox_inches='tight', dpi=150)
                    plt.close(fig1)
                    st.pyplot(fig1)

                with col2:
                    st.subheader(translate("Prediction and Risk Breakdown"))
                    fig2 = plot_pie(result, risk_score)
                    pie_chart_path = f"pie_chart_{patient_id}.png"
                    fig2.savefig(pie_chart_path, bbox_inches='tight', dpi=150)
                    plt.close(fig2)
                    st.pyplot(fig2)

                st.subheader(translate("Feature Importance"))
                fig3, feature_importance_df = plot_feature_importance()
                bar_chart_path = f"bar_chart_{patient_id}.png"
                if fig3 is not None and feature_importance_df is not None:
                    fig3.savefig(bar_chart_path, bbox_inches='tight', dpi=150)
                    plt.close(fig3)
                    st.pyplot(fig3)
                    st.write(translate("Top 5 Important Features:"))
                    st.write(feature_importance_df)
                else:
                    st.warning(translate("Feature importance graph could not be generated. Please check model training or data."))

                st.subheader(translate("Download PDF Report"))
                # Retrieve all prediction data from session state for PDF generation
                prediction_data = st.session_state.get('prediction_data', {})
                if not prediction_data:
                    st.error(translate("No prediction data available. Please make a prediction first."))
                else:
                    pdf_numerical_features = prediction_data.get('numerical_features', [])
                    pdf_new_features = prediction_data.get('new_features', [])
                    pdf_patient_id = prediction_data.get('patient_id', patient_id)
                    pdf_result = prediction_data.get('result', result)
                    pdf_risk_score = prediction_data.get('risk_score', risk_score)

                    pdf_file = generate_pdf_report(pdf_patient_id, pdf_result, pdf_numerical_features, pdf_new_features, pdf_risk_score)
                    with open(pdf_file, "rb") as file:
                        st.download_button(
                            label=translate("Download Report"),
                            data=file,
                            file_name=pdf_file,
                            mime="application/pdf",
                            key="download_button"
                        )

                # Save patient history using session state data
                save_patient_history(pdf_patient_id, pdf_result, pdf_risk_score, pdf_numerical_features, pdf_new_features)
                st.write(translate("Prediction and features saved to patient history."))

                st.subheader(translate("Recommended Next Steps"))
                next_steps = [
                    "Immediate biopsy or further imaging (e.g., MRI, ultrasound) to confirm malignancy",
                    "Consult an oncologist for treatment planning (e.g., surgery, chemotherapy, radiation)",
                    "Monitor for metastasis with regular scans"
                ] if pdf_result == "Malignant" else [
                    "Continue regular screenings (e.g., annual mammograms)",
                    "Maintain healthy lifestyle habits to reduce risk",
                    "Follow up with a doctor if new symptoms appear"
                ]
                translated_steps = translate(next_steps)  # Translate each step individually
                for step in translated_steps:
                    st.write(step)

                email = st.text_input(translate("Enter your email for results (optional)"))
                if st.button(translate("Send Results via Email")) and email:
                    send_email(email, pdf_patient_id, pdf_result, pdf_risk_score, pdf_file)

            except Exception as e:
                st.error(f"Prediction error: {e}")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(translate("Patient History"))
    with col2:
        if st.button(translate("View")):
            history_df = get_patient_history()
            if not history_df.empty:
                st.subheader(translate("Full Patient History"))
                st.write(history_df)
            else:
                st.write(translate("No patient history available."))

    if patient_id:
        history = get_patient_history(patient_id)
        if not history.empty:
            st.write(translate("Recent History for This Patient:"))
            st.write(history)
        else:
            st.write(translate("No history found for this patient."))

    # Chatbot Section
    st.header(translate("Ask About Breast Cancer"))
    st.write(translate("Interact with PinkScan Bot for instant answers about breast cancer. Ask anything!"))

    # Chat history and input
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    st.subheader(translate("Chat History"))
    chatbot_container = st.container()
    with chatbot_container:
        for message in st.session_state.chat_history:
            username = "You" if message["user"] else "PinkScan Bot"
            message_class = "chatbot-user" if message["user"] else "chatbot-bot"
            st.markdown(f'<div class="chatbot-message {message_class}"><strong style="color: #ff69b4;">{username}:</strong> {message["text"]}</div>', unsafe_allow_html=True)

    # Chat input
    user_input = st.text_input(translate("Type your question here"), key="chat_input", help="Ask any question about breast cancer")
    if st.button(translate("Send"), key="chat_send"):
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"text": user_input, "user": True})
            # Get chatbot response
            response = get_chatbot_response(user_input)
            # Add chatbot response to chat history
            st.session_state.chat_history.append({"text": response, "user": False})
            # Clear input and rerun
            st.rerun()

    st.header(translate("Batch Prediction"))
    uploaded_file = st.file_uploader(translate("Upload a CSV file with feature values"), type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Normalize column names: strip quotes, extra spaces, and standardize case
            df.columns = [col.strip("'").strip().title() for col in df.columns]
            # Convert expected feature names to title case for consistency
            all_feature_names = [col.title() for col in numerical_feature_names + new_feature_names]
            st.write(translate("Uploaded Data (Normalized Column Names):"))
            st.write(df)

            if "Patient Id" not in df.columns:  # Check for title case
                st.error(translate("The uploaded file must contain a 'Patient ID' column (case-insensitive)."))
            else:
                missing_columns = [col for col in all_feature_names if col not in df.columns]
                if missing_columns:
                    st.error(translate(f"The uploaded file is missing the following required columns: {missing_columns}"))
                    st.write(translate("Required columns (case-normalized):"))
                    st.write(all_feature_names)
                    st.write(translate("Actual columns in uploaded file:"))
                    st.write(list(df.columns))
                else:
                    # Check for non-numeric values or NaNs in numerical features
                    numerical_cols = [col for col in numerical_feature_names if col in df.columns]
                    if not df[numerical_cols].apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all()).all():
                        st.error(translate("All numerical feature values must be numeric. Please check the CSV for invalid entries."))
                    else:
                        # Encode all 37 features for prediction
                        predictions = []
                        risk_scores = []
                        for _, row in df.iterrows():
                            numerical_data = [float(row[col]) for col in numerical_feature_names]
                            numerical_data = np.array(numerical_data, dtype=float).flatten()  # Ensure 1D
                            new_data = [row[col] for col in new_feature_names]
                            encoded_features = encode_features(numerical_data, new_data)
                            pred, prob = predict_breast_cancer(encoded_features)
                            predictions.append(pred)
                            risk_scores.append(prob)

                        df["Prediction"] = predictions
                        df["Risk Score (%)"] = risk_scores
                        df["Risk Level"] = ["Low" if r < 33.33 else "Medium" if r < 66.67 else "High" for r in risk_scores]
                        st.write(translate("Predictions:"))
                        st.write(df)
                        # Option to download batch results as CSV
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label=translate("Download Batch Predictions"),
                            data=csv,
                            file_name="batch_predictions.csv",
                            mime="text/csv",
                            key="batch_download"
                        )
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.write(translate("Please ensure the CSV is properly formatted with all required columns."))

    with st.form(translate("feedback_form")):
        feedback = st.text_area(translate("Provide feedback or suggestions"))
        if st.form_submit_button(translate("Submit Feedback")):
            with open("feedback.txt", "a") as f:
                f.write(f"Patient ID: {patient_id}, Feedback: {feedback}, Date: {pd.Timestamp.now()}\n")
            st.success(translate("Thank you for your feedback!"))

# Improved graph functions
@st.cache_data
def plot_histogram(numerical_features, result):
    numerical_features = np.array(numerical_features, dtype=float).flatten()
    if numerical_features.ndim != 1 or len(numerical_features) != 30:
        st.error(f"Numerical features shape mismatch: expected (30,), got {numerical_features.shape}")
        raise ValueError("Numerical features must be a 1D array with 30 values.")
    
    data = pd.DataFrame([numerical_features], columns=numerical_feature_names).T
    data.columns = ['Value']
    data['Diagnosis'] = 'Malignant' if result == "Malignant" else 'Benign'
    
    fig, axes = plt.subplots(6, 5, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, feature in enumerate(numerical_feature_names):
        ax = axes[i]
        value = data.loc[feature, 'Value']
        min_val, max_val = numerical_feature_ranges[feature]
        reference_range = [min_val, (min_val + max_val) / 2, max_val]  # Simple range for reference
        
        # Plot box plot with current value as a point
        sns.boxplot(data=[value], ax=ax, color="#ff69b4", width=0.3, showfliers=False)
        ax.plot(0, value, 'ro', label='Current Value')  # Red dot for current value
        ax.set_title(feature, fontsize=10, color="#ff69b4")
        ax.set_xlabel("Value", fontsize=8, color="#333333")
        ax.set_ylabel("Frequency", fontsize=8, color="#333333")
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=8, loc="upper right")
    
    for j in range(len(numerical_feature_names), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    return fig

@st.cache_data
def plot_pie(result, risk_score):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [translate("Malignant"), translate("Benign"), translate("Non-Malignant")]
    sizes = [risk_score if result == "Malignant" else 0, 
             100 - risk_score if result == "Benign" else 0, 
             100 - risk_score if result == "Malignant" else 100 - risk_score]
    colors = ["#FF6347", "#4682B4", "#32CD32"]
    explode = (0.1 if result == "Malignant" else 0, 0.1 if result == "Benign" else 0, 0)
    active_labels = [l for l, s in zip(labels, sizes) if s > 0]
    active_sizes = [s for s in sizes if s > 0]
    active_colors = [c for c, s in zip(colors, sizes) if s > 0]
    active_explode = [e for e, s in zip(explode, sizes) if s > 0]
    wedges, texts, autotexts = ax.pie(active_sizes, explode=active_explode, labels=active_labels, 
                                     colors=active_colors, autopct=lambda pct: f'{pct:.1f}%',
                                     shadow=True, startangle=90, textprops={'fontsize': 10, 'color': 'black'})
    ax.axis("equal")
    plt.setp(autotexts, size=10, weight="bold", color="white")
    plt.title("Prediction and Risk Breakdown", fontsize=12, color="#ff69b4")
    plt.tight_layout()
    return fig

@st.cache_data
def plot_feature_importance():
    importances = model.feature_importances_
    if len(importances) != 37:  # Ensure model is trained on 37 features
        st.error(f"Model feature importances length mismatch. Expected 37, got {len(importances)}. Please retrain the model with 37 features.")
        return None, None
    
    all_feature_names = numerical_feature_names + new_feature_names
    feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
    
    # Normalize importances to sum to 1 for better visualization
    total_importance = importances.sum()
    if total_importance == 0:
        st.warning("Feature importances are zero. Please check model training or retrain with more varied data.")
        normalized_importances = np.ones(len(importances)) / len(importances)  # Equal importance for all features
    else:
        normalized_importances = importances / total_importance
    
    feature_importance_df['importance'] = normalized_importances
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(5)
    
    if feature_importance_df.empty or feature_importance_df['importance'].sum() == 0:
        st.warning("No significant feature importances found. Please check model or data.")
        return None, None
    
    fig, ax = plt.subplots(figsize=(8, 6))  # Increased size for better visibility
    sns.barplot(x='importance', y='feature', data=feature_importance_df, ax=ax, color="#ff69b4", edgecolor="black")
    
    # Customize the plot for better readability and accuracy
    ax.set_title("Top 5 Most Important Features", fontsize=14, color="#ff69b4", pad=15)
    ax.set_xlabel("Importance (Normalized)", fontsize=12, color="#333333")
    ax.set_ylabel("Feature", fontsize=12, color="#333333")
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add value labels on top of bars for clarity
    for i, v in enumerate(feature_importance_df['importance']):
        ax.text(v, i, f'{v:.3f}', color='black', ha="left", va="center", fontsize=8)
    
    # Ensure the x-axis starts at 0 (non-negative importances)
    ax.set_xlim(0, None)
    ax.grid(True, linestyle='--', alpha=0.7, axis='x')
    plt.tight_layout()
    
    return fig, feature_importance_df

# Simple rule-based chatbot responses
def get_chatbot_response(user_input):
    user_input = user_input.lower().strip()
    responses = {
        "hello": "Hello! How can I assist you with breast cancer information today?",
        "hi": "Hi there! What would you like to know about breast cancer?",
        "treatment": "Common treatments include surgery, radiation, chemotherapy, and hormone therapy. Please consult your doctor for personalized advice.",
        "prevention": "Prevention includes regular screenings, maintaining a healthy diet, exercising regularly, and avoiding smoking/alcohol.",
        "symptoms": "Symptoms include a lump in the breast or underarm, changes in breast size/shape, nipple discharge, or skin redness/pitting. Consult a doctor if you notice these.",
        "cure": "There’s no universal cure, but early detection and treatment (e.g., surgery, therapy) can manage or eliminate breast cancer. Consult a healthcare professional.",
        "what is breast cancer": "Breast cancer is a type of cancer that forms in the cells of the breasts, often as a lump or tumor. It’s common among women but can affect men too.",
        "help": "I can help with information on breast cancer symptoms, prevention, treatment, and more. What would you like to know?",
        "default": "I'm sorry, I don’t understand that. Please ask about breast cancer symptoms, prevention, treatment, or say 'help' for options."
    }
    for keyword in responses:
        if keyword in user_input:
            return responses[keyword]
    return responses["default"]

if __name__ == "__main__":
    main()