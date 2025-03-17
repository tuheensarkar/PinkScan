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
import re
import time

# Set page config as the first Streamlit command
st.set_page_config(page_title="PinkScan - Predict Tool", page_icon=":hospital:", layout="wide")

# Cache model loading
@st.cache_resource
def load_model():
    try:
        return joblib.load('breast_cancer_model.pkl')
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'breast_cancer_model.pkl' is in the correct directory.")
        st.stop()

# Load model
model = load_model()

# Healthy profile (37 features)
healthy_profile = [
    12.0, 18.0, 78.0, 450.0, 0.09, 0.07, 0.03, 0.02, 0.16, 0.06,  # Mean Features
    0.4, 1.0, 2.5, 20.0, 0.005, 0.01, 0.015, 0.005, 0.015, 0.002,  # Standard Error Features
    13.5, 22.0, 88.0, 550.0, 0.11, 0.14, 0.08, 0.04, 0.22, 0.07,  # Worst Features
    0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.7  # Additional Features
]

# Feature names (37 features)
feature_names = [
    # Mean Features (10)
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
    "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    
    # Standard Error Features (10)
    "Radius Error", "Texture Error", "Perimeter Error", "Area Error", "Smoothness Error",
    "Compactness Error", "Concavity Error", "Concave Points Error", "Symmetry Error", "Fractal Dimension Error",
    
    # Worst Features (10)
    "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness",
    "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension",
    
    # Additional Features (7)
    "Tumor Size", "Lymph Node Status", "Mitosis Rate", "Nuclear Pleomorphism", "Tubule Formation",
    "Nuclear Grade", "Histological Grade"
]

# Feature ranges (37 features)
feature_ranges = {
    # Mean Features (10)
    "Mean Radius": (6.0, 28.0), "Mean Texture": (9.0, 39.0), "Mean Perimeter": (43.0, 188.0), "Mean Area": (143.0, 2501.0),
    "Mean Smoothness": (0.05, 0.16), "Mean Compactness": (0.02, 0.35), "Mean Concavity": (0.0, 0.43),
    "Mean Concave Points": (0.0, 0.2), "Mean Symmetry": (0.1, 0.3), "Mean Fractal Dimension": (0.05, 0.1),
    
    # Standard Error Features (10)
    "Radius Error": (0.1, 2.5), "Texture Error": (0.36, 4.9), "Perimeter Error": (0.76, 21.98),
    "Area Error": (6.8, 542.2), "Smoothness Error": (0.001, 0.03), "Compactness Error": (0.002, 0.14),
    "Concavity Error": (0.0, 0.4), "Concave Points Error": (0.0, 0.05), "Symmetry Error": (0.008, 0.08),
    "Fractal Dimension Error": (0.001, 0.03),
    
    # Worst Features (10)
    "Worst Radius": (7.9, 36.04), "Worst Texture": (12.02, 49.54), "Worst Perimeter": (50.41, 251.2),
    "Worst Area": (185.2, 4254.0), "Worst Smoothness": (0.07, 0.22), "Worst Compactness": (0.02, 1.06),
    "Worst Concavity": (0.0, 1.25), "Worst Concave Points": (0.0, 0.29), "Worst Symmetry": (0.16, 0.66),
    "Worst Fractal Dimension": (0.06, 0.21),
    
    # Additional Features (7)
    "Tumor Size": (0.0, 5.0), "Lymph Node Status": (0.0, 3.0), "Mitosis Rate": (0.0, 1.0),
    "Nuclear Pleomorphism": (0.0, 3.0), "Tubule Formation": (0.0, 3.0), "Nuclear Grade": (0.0, 3.0),
    "Histological Grade": (0.0, 3.0)
}

LANGUAGES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'bn': 'Bengali', 'hi': 'Hindi'
}

@st.cache_data
def translate_text(text, lang):
    if lang == 'en':
        return text  # No translation needed for English
    translator = GoogleTranslator(source='auto', target=lang)
    try:
        return translator.translate(text)
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

def translate(_text):
    language = st.session_state.get('language', 'English')
    return translate_text(_text, list(LANGUAGES.keys())[list(LANGUAGES.values()).index(language)])

@st.cache_data
def predict_breast_cancer(input_features):
    if len(input_features) != 37:
        raise ValueError(f"Expected 37 features, but got {len(input_features)} features.")
    
    input_features = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_features)
    probability = model.predict_proba(input_features)[0]
    return "Malignant" if prediction[0] == 0 else "Benign", probability[0] * 100

# PDF Report Generation
def generate_pdf_report(patient_id, result, input_features, healthy_profile, risk_score):
    pdf = FPDF()
    pdf.add_page()
    
    # Set consistent margins for all pages
    left_margin = 20
    top_margin = 20
    right_margin = 20
    pdf.set_margins(left=left_margin, top=top_margin, right=right_margin)
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # Border
    pdf.set_line_width(0.5)
    pdf.rect(10, 10, 190, 277)
    
    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Breast Cancer Prediction Report", 0, 1, "C")
    pdf.ln(5)
    
    # Patient Info
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, f"Patient ID: {patient_id}", 0, 1)
    pdf.cell(0, 8, f"Diagnosis: {result}", 0, 1)
    pdf.cell(0, 8, f"Risk Score: {risk_score:.2f}% likelihood of malignant tumor", 0, 1)
    pdf.ln(10)
    
    # Feature Comparison Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Feature Comparison", 0, 1)
    pdf.set_font("Helvetica", size=10)
    
    col_widths = [60, 50, 50]
    headers = ["Feature Name", "Patient Value", "Healthy Value"]
    pdf.set_fill_color(220, 220, 220)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, 1, 0, "C", 1)
    pdf.ln(8)
    
    pdf.set_fill_color(255, 255, 255)
    for i, (name, patient_val, healthy_val) in enumerate(zip(feature_names, input_features, healthy_profile)):
        pdf.cell(col_widths[0], 8, name, 1)
        pdf.cell(col_widths[1], 8, f"{patient_val:.2f}", 1, 0, "C")
        pdf.cell(col_widths[2], 8, f"{healthy_val:.2f}", 1, 0, "C")
        pdf.ln(8)
    
    # Move to second page
    pdf.add_page()
    
    # Doctor's Recommendations
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Doctor's Recommendations", 0, 1)
    pdf.set_font("Helvetica", size=10)
    tips = [
        "1. Schedule regular mammograms as recommended by your doctor",
        "2. Perform monthly breast self-examinations",
        "3. Maintain a healthy weight through diet and exercise",
        "4. Limit alcohol consumption",
        "5. Quit smoking and avoid secondhand smoke"
    ]
    for tip in tips:
        pdf.cell(0, 6, tip.encode('latin-1', 'replace').decode('latin-1'), 0, 1)
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Recommended Dietary Routine", 0, 1)
    pdf.set_font("Helvetica", size=10)
    diet = [
        "1. Eat plenty of fruits and vegetables (5+ servings daily)",
        "2. Choose whole grains over refined grains",
        "3. Include lean proteins (fish, poultry, beans)",
        "4. Limit processed and red meats",
        "5. Use healthy fats (olive oil, avocados, nuts)"
    ]
    for item in diet:
        pdf.cell(0, 6, item.encode('latin-1', 'replace').decode('latin-1'), 0, 1)
    
    # Add space before pie chart
    pdf.ln(10)
    
    # Pie Chart Generation with Dynamic Healthy Comparison
    pie_chart_path = f"pie_chart_{patient_id}.png"
    
    try:
        # Calculate a simple "Healthy" score (e.g., similarity to healthy_profile)
        healthy_similarity = 100 - np.mean(np.abs(np.array(input_features) - np.array(healthy_profile)) / 
                                          (np.array([r[1] - r[0] for r in feature_ranges.values()]))) * 100
        healthy_similarity = max(0, min(healthy_similarity, 20))  # Cap at 20% for visualization

        # Enhanced Pie Chart
        labels = ["Malignant Risk", "Benign Probability", "Healthy Similarity"]
        sizes = [risk_score, 100 - risk_score - healthy_similarity, healthy_similarity]
        colors = ["#FF4040", "#40C4FF", "#D3D3D3"]  # Red (Malignant), Light Blue (Benign), Gray (Healthy)
        explode = (0.1 if result == "Malignant" else 0, 0.1 if result == "Benign" else 0, 0)  # Explode predicted slice
        
        fig, ax = plt.subplots(figsize=(6, 4))  # Slightly larger for clarity
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=None, colors=colors, autopct='%1.1f%%',
                                          shadow=True, startangle=90, textprops={'fontsize': 10, 'color': 'black'})
        ax.axis("equal")
        
        # Customize text
        plt.setp(autotexts, size=10, weight="bold", color="white")
        plt.setp(texts, size=10)
        
        # Add legend
        ax.legend(wedges, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)
        
        # Title
        plt.title("Prediction Breakdown", fontsize=12, pad=20)
        plt.tight_layout()
        
        # Save with higher DPI for better PDF quality
        fig.savefig(pie_chart_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        # Add pie chart to PDF
        y_position = pdf.get_y() + 10  # Add padding after dietary routine
        pdf.image(pie_chart_path, x=left_margin, y=y_position, w=90)  # Slightly larger width
    except Exception as e:
        st.error(f"Error generating pie chart for PDF: {e}")
    finally:
        if os.path.exists(pie_chart_path):
            try:
                os.remove(pie_chart_path)
            except OSError as e:
                st.warning(f"Could not remove temporary file {pie_chart_path}: {e}")

    pdf_file = f"report_{patient_id}.pdf"
    pdf.output(pdf_file)
    return pdf_file

# Email Functionality
def send_email(email, patient_id, result, risk_score, pdf_file):
    sender_email = "your_gmail_address@gmail.com"
    app_password = "your_app_password"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = f"Breast Cancer Prediction Report for {patient_id}"
    msg.attach(MIMEText(f"Prediction: {result}\nRisk Score: {risk_score:.2f}% likelihood of malignant tumor"))

    try:
        with open(pdf_file, "rb") as f:
            part = MIMEApplication(f.read(), Name=pdf_file)
            part['Content-Disposition'] = f'attachment; filename="{pdf_file}"'
            msg.attach(part)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.send_message(msg)
        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Error sending email: {e}")

# Chatbot for FAQs
def chatbot():
    st.sidebar.header("Chatbot - Ask About Breast Cancer")
    question = st.sidebar.text_input("Type your question here")
    if question:
        q_lower = question.lower()
        if "treatment" in q_lower:
            st.sidebar.write("Common treatments include surgery, radiation, chemotherapy, and hormone therapy. Consult your doctor for personalized advice.")
        elif "prevention" in q_lower:
            st.sidebar.write("Prevention includes regular screenings, healthy diet, exercise, and avoiding smoking/alcohol.")
        elif "symptoms" in q_lower:
            st.sidebar.write("Symptoms include a lump, changes in breast size/shape, nipple discharge, or skin redness/pitting.")
        elif "risk factors" in q_lower:
            st.sidebar.write("Risk factors include age, family history, genetic mutations, obesity, and alcohol consumption.")
        elif "diagnosis" in q_lower:
            st.sidebar.write("Diagnosis typically involves mammograms, ultrasounds, biopsies, and MRI scans.")
        else:
            st.sidebar.write("I'm sorry, I don't have information on that. Please consult a healthcare professional.")

# Feedback Collection
def feedback_form(patient_id):
    with st.form("feedback_form"):
        st.header("Feedback Form")
        feedback = st.text_area("Provide feedback or suggestions")
        if st.form_submit_button("Submit Feedback"):
            with open("feedback.txt", "a") as f:
                f.write(f"Patient ID: {patient_id}, Feedback: {feedback}, Date: {pd.Timestamp.now()}\n")
            st.success("Thank you for your feedback!")

# Main Function
def main():
    # Custom CSS for PinkScan branding
    st.markdown("""
    <style>
    .stApp {
        background-color: #fff;
        color: #333;
    }
    .stButton>button {
        background-color: #ff69b4;
        color: white;
        border-radius: 25px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff4d9e;
    }
    .stHeader {
        color: #ff69b4;
        font-size: 2.5rem;
        text-align: center;
        font-weight: bold;
    }
    .stSubheader {
        color: #333;
        font-size: 1.5rem;
    }
    .stText {
        color: #666;
    }
    .stWarning {
        background-color: #ffebee;
        border: 1px solid #ffcdd2;
        border-radius: 5px;
        padding: 10px;
    }
    .stSuccess {
        background-color: #e8f5e9;
        border: 1px solid #c8e6c9;
        border-radius: 5px;
        padding: 10px;
    }
    .stError {
        background-color: #ffebee;
        border: 1px solid #ffcdd2;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    if 'language' not in st.session_state:
        st.session_state.language = 'English'
    language = st.sidebar.selectbox("Select Language", options=list(LANGUAGES.values()), 
                                  index=list(LANGUAGES.values()).index(st.session_state.language))
    st.session_state.language = language

    st.title(translate("PinkScan - Breast Cancer Prediction Tool"))
    st.write(translate("""
        Stay ahead with early detection and smart decisions. This AI-powered tool predicts whether a breast tumor is *malignant* or *benign* based on input features and compares it with a typical healthy (benign) profile.
    """))

    st.sidebar.header(translate("About PinkScan"))
    st.sidebar.write(translate("""
        PinkScan uses a *Random Forest Classifier* trained on the Breast Cancer Wisconsin (Diagnostic) Dataset.
        - *Malignant*: Cancerous tumor.
        - *Benign*: Non-cancerous tumor.
    """))

    with st.sidebar.expander(translate("Learn More About Breast Cancer")):
        st.markdown(f"""
            <h2 style='color: #ff69b4;'>{translate('Understanding Breast Cancer')}</h2>
            <p style='color: #333;'>
                {translate("Breast cancer is a type of cancer that develops in the cells of the breasts. It is the most common cancer among women worldwide, but it can also affect men.")}
            </p>
            <h3 style='color: #ff69b4;'>{translate('Symptoms')}</h3>
            <ul>
                <li>{translate("A lump or thickening in the breast or underarm.")}</li>
                <li>{translate("Changes in the size, shape, or appearance of the breast.")}</li>
                <li>{translate("Nipple discharge or inversion.")}</li>
                <li>{translate("Redness or pitting of the breast skin (like an orange peel).")}</li>
            </ul>
            <h3 style='color: #ff69b4;'>{translate('Prevention')}</h3>
            <ul>
                <li>{translate("Maintain a healthy weight.")}</li>
                <li>{translate("Exercise regularly.")}</li>
                <li>{translate("Limit alcohol consumption.")}</li>
                <li>{translate("Avoid smoking.")}</li>
            </ul>
        """, unsafe_allow_html=True)

    patient_id = st.text_input(translate("Enter Patient ID"), placeholder="e.g., PAT12345")

    feature_groups = {
        translate("Mean Features"): feature_names[:10],
        translate("Standard Error Features"): feature_names[10:20],
        translate("Worst Features"): feature_names[20:30],
        translate("Clinical Features"): feature_names[30:]  # Add group for additional features
    }

    input_features = []
    for group_name, features in feature_groups.items():
        with st.expander(f"{group_name}"):
            for feature in features:
                min_val, max_val = feature_ranges[feature]
                value = st.number_input(f"{feature}", min_value=float(min_val), max_value=float(max_val), 
                                      value=float(min_val), help=f"Enter a value between {min_val} and {max_val}")
                if value < min_val or value > max_val:
                    st.warning(f"Value for {feature} is outside typical range ({min_val}-{max_val}).")
                input_features.append(value)

    if st.button(translate("Predict")):
        if not patient_id:
            st.warning(translate("Please enter a Patient ID."))
        else:
            try:
                result, risk_score = predict_breast_cancer(input_features)
                if result == "Malignant":
                    st.error(f"Patient ID: {patient_id}\n{translate('Prediction')}: {result} 🚨")
                else:
                    st.success(f"Patient ID: {patient_id}\n{translate('Prediction')}: {result} ✅")

                st.write(f"{translate('Risk Score')}: {risk_score:.2f}% {translate('likelihood of malignant tumor')}")

                # Display Pie Chart in Streamlit App
                st.subheader(translate("Prediction Breakdown"))

                # Calculate Healthy Similarity
                healthy_similarity = 100 - np.mean(np.abs(np.array(input_features) - np.array(healthy_profile)) / 
                                                  (np.array([r[1] - r[0] for r in feature_ranges.values()]))) * 100
                healthy_similarity = max(0, min(healthy_similarity, 20))  # Cap at 20% for visualization

                labels = ["Malignant Risk", "Benign Probability", "Healthy Similarity"]
                sizes = [risk_score, 100 - risk_score - healthy_similarity, healthy_similarity]
                colors = ["#FF4040", "#40C4FF", "#D3D3D3"]  # Red (Malignant), Light Blue (Benign), Gray (Healthy)
                explode = (0.1 if result == "Malignant" else 0, 0.1 if result == "Benign" else 0, 0)  # Explode predicted slice

                fig, ax = plt.subplots(figsize=(6, 4))
                wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=None, colors=colors, autopct='%1.1f%%',
                                                  shadow=True, startangle=90, textprops={'fontsize': 10, 'color': 'black'})
                ax.axis("equal")
                plt.setp(autotexts, size=10, weight="bold", color="white")
                plt.setp(texts, size=10)

                # Add legend
                ax.legend(wedges, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)

                plt.title("Prediction Breakdown", fontsize=12, pad=20)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Generate PDF Report
                with st.spinner("Generating PDF Report..."):
                    pdf_file = generate_pdf_report(patient_id, result, input_features, healthy_profile, risk_score)
                    with open(pdf_file, "rb") as file:
                        st.download_button(
                            label=translate("Download Report"),
                            data=file,
                            file_name=pdf_file,
                            mime="application/pdf",
                            key="download_button"
                        )

                # Email Functionality
                email = st.text_input(translate("Enter your email for results (optional)"))
                if st.button(translate("Send Results via Email")) and email:
                    if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                        send_email(email, patient_id, result, risk_score, pdf_file)
                    else:
                        st.error("Invalid email address. Please enter a valid email.")

            except Exception as e:
                st.error(f"Prediction error: {e}")

    # Chatbot for FAQs
    chatbot()

    # Feedback Form
    if patient_id:
        feedback_form(patient_id)

    # Reset Button
    if st.button(translate("Reset Form")):
        st.session_state.clear()
        st.experimental_rerun()

if __name__ == "__main__":
    main()
