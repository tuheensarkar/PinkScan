import streamlit as st
import joblib
import numpy as np
import pandas as pd
from deep_translator import GoogleTranslator
from fpdf import FPDF
import os
import logging
import tempfile

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Streamlit page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="PinkScan - Patient Self-Test",
    page_icon=":hospital:",
    layout="wide"
)

# Custom CSS for UI enhancement
st.markdown("""
    <style>
        /* General Styling */
        .stApp {
            background: linear-gradient(135deg, #fff, #fce4ec);
            font-family: 'Poppins', sans-serif;
            color: #333333;
        }
        h1, h2, h3 {
            color: #ff69b4;
            font-weight: 700;
            text-shadow: 0 2px 10px rgba(255, 105, 180, 0.2);
        }
        h1 {
            font-size: 3.5rem;
            margin-bottom: 25px;
            text-align: center;
        }
        h2 {
            font-size: 2.5rem;
        }
        p, label, .stTextInput label, .stNumberInput label, .stSelectbox label {
            color: #555555;
            font-size: 1.2rem;
            font-weight: 300;
            line-height: 1.6;
        }

        /* Sidebar */
        .css-1d391kg {
            background: rgba(255, 255, 255, 0.95);
            box-shadow: 0 5px 20px rgba(255, 105, 180, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(5px);
        }
        .st-expander {
            background: #fff;
            border: 1px solid #ffb6c1;
            border-radius: 10px;
        }
        .st-expander summary {
            color: #ff69b4;
            font-weight: 600;
        }

        /* Inputs */
        .stTextInput input, .stNumberInput input, .stSelectbox select {
            border: 2px solid #ffb6c1;
            border-radius: 12px;
            padding: 10px;
            font-size: 1.2rem;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }
        .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {
            border-color: #ff69b4;
            box-shadow: 0 0 12px rgba(255, 105, 180, 0.3);
            outline: none;
        }

        /* Buttons */
        .stButton button {
            background: linear-gradient(45deg, #ff69b4, #ff4d9e);
            color: #fff;
            border: none;
            border-radius: 30px;
            padding: 12px 40px;
            font-size: 1.3rem;
            font-weight: 700;
            transition: all 0.4s ease;
        }
        .stButton button:hover {
            background: linear-gradient(45deg, #ff4d9e, #ff69b4);
            transform: scale(1.05);
            box-shadow: 0 15px 30px rgba(255, 77, 158, 0.5);
        }

        /* Prediction Output */
        .prediction-box {
            background: linear-gradient(135deg, #fff, #f9f2f5);
            border: 2px solid #ffb6c1;
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
            margin-top: 20px;
        }
        .prediction-box p {
            font-size: 1.4rem;
            color: #333333;
            font-weight: 400;
        }
        .prediction-box span {
            color: #ff69b4;
            font-weight: 600;
        }

        /* Chatbot */
        .chatbot-container {
            border: 2px solid #ff69b4;
            border-radius: 15px;
            padding: 20px;
            background: #fff;
            margin-top: 30px;
            box-shadow: 0 5px 20px rgba(255, 105, 180, 0.1);
        }
        .chatbot-message {
            margin: 10px 0;
            padding: 15px;
            border-radius: 10px;
            background: #f8f9fa;
            color: #333333;
        }
        .chatbot-user {
            background: #e9ecef;
            text-align: right;
        }
        .chatbot-bot {
            background: #fff;
            text-align: left;
            border: 1px solid #ffb6c1;
        }
        .chatbot-message strong {
            color: #ff69b4;
            font-weight: 700;
        }
    </style>
""", unsafe_allow_html=True)

# Language selection dictionary
LANGUAGES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'bn': 'Bengali', 'hi': 'Hindi'
}

# Translation function
@st.cache_data
def translate_text(text, lang):
    translator = GoogleTranslator(source='auto', target=lang)
    try:
        if isinstance(text, list):
            return [translator.translate(item) for item in text if isinstance(item, str) and len(item) <= 5000]
        return translator.translate(text) if isinstance(text, str) and len(text) <= 5000 else text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

def translate(_text):
    language = st.session_state.get('language', 'English')
    return translate_text(_text, list(LANGUAGES.keys())[list(LANGUAGES.values()).index(language)])

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('patient_self_test_model.pkl')
        scaler = joblib.load('scaler.pkl')
        logger.info("Model and scaler loaded successfully")
        return model, scaler
    except FileNotFoundError as e:
        st.error("Model or scaler file not found. Please ensure 'patient_self_test_model.pkl' and 'scaler.pkl' are present.")
        logger.error(f"FileNotFoundError: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        logger.error(f"Exception: {e}")
        return None, None

model, scaler = load_model_and_scaler()

# Feature definitions
feature_names = [
    "Age", "Family History of Breast Cancer", "Tumor Size (mm)", "Lymph Node Involvement",
    "Menopausal Status", "Breast Pain", "Skin Changes"
]

feature_options = {
    "Family History of Breast Cancer": ["Yes", "No", "Unknown"],
    "Lymph Node Involvement": ["Yes", "No", "Unknown"],
    "Menopausal Status": ["Pre-menopausal", "Post-menopausal", "Unknown"],
    "Breast Pain": ["No", "Yes"],
    "Skin Changes": ["No", "Yes"]
}

feature_ranges = {
    "Age": (18.0, 100.0),
    "Tumor Size (mm)": (0.0, 100.0)
}

# Feature encoding for consistency with training data
def encode_features(features):
    feature_mapping = {
        "Family History of Breast Cancer": {"No": 0, "Yes": 1, "Unknown": 0.5},
        "Lymph Node Involvement": {"No": 0, "Yes": 1, "Unknown": 0.5},
        "Menopausal Status": {"Pre-menopausal": 0, "Post-menopausal": 1, "Unknown": 0.5},
        "Breast Pain": {"No": 0, "Yes": 1},
        "Skin Changes": {"No": 0, "Yes": 1}
    }
    encoded = []
    for i, feature in enumerate(features):
        if feature_names[i] in feature_mapping:
            encoded.append(feature_mapping[feature_names[i]][feature])
        else:
            min_val, max_val = feature_ranges[feature_names[i]]
            value = float(feature)
            if not (min_val <= value <= max_val):
                st.warning(f"{feature_names[i]} value {value} is outside typical range ({min_val}-{max_val}).")
            encoded.append(value)
    return np.array(encoded)

# Prediction function with reversed logic
def predict_breast_cancer(features):
    if model is None or scaler is None:
        logger.error("Model or scaler is None")
        return "Error", 0.0
    try:
        encoded_features = encode_features(features)
        if len(encoded_features) != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features, got {len(encoded_features)}")
        scaled_features = scaler.transform(encoded_features.reshape(1, -1))
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0]
        # Reversed logic: 0 = Malignant, 1 = Benign
        result = "Malignant" if prediction == 0 else "Benign"
        risk_score = probability[0] * 100 if prediction == 0 else probability[1] * 100
        logger.info(f"Encoded features: {encoded_features}, Prediction: {result}, Risk Score: {risk_score}")
        return result, risk_score
    except Exception as e:
        st.error(f"Prediction error: {e}")
        logger.error(f"Prediction error: {e}")
        return "Error", 0.0

# PDF generation function
def generate_pdf_report(patient_id, result, features, risk_score):
    pdf = FPDF()
    pdf.add_page()
    
    # Page layout with pink border
    margin = 20
    page_width = 210 - (2 * margin)
    pdf.set_margins(left=margin, top=margin, right=margin)
    pdf.set_line_width(0.5)
    pdf.set_draw_color(255, 105, 180)  # Pink (#ff69b4)
    pdf.rect(margin, margin, page_width, 257)
    
    # Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 105, 180)
    pdf.cell(0, 10, translate("PinkScan - Self-Test Report"), 0, 1, "C")
    pdf.ln(5)
    
    # Patient Information
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, translate("Patient Information"), 0, 1)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, f"{translate('Patient ID')}: {patient_id}", 0, 1)
    pdf.cell(0, 8, f"{translate('Report Date')}: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(5)
    
    # Prediction Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, translate("Prediction Summary"), 0, 1)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, f"{translate('Diagnosis')}: {translate(result)}", 0, 1)
    pdf.cell(0, 8, f"{translate('Risk Score')}: {risk_score:.2f}% {translate('likelihood of malignancy')}", 0, 1)
    pdf.ln(10)
    
    # Features Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, translate("Input Features"), 0, 1)
    pdf.set_font("Helvetica", size=10)
    col_widths = [page_width * 0.6, page_width * 0.4]
    headers = [translate("Feature"), translate("Value")]
    pdf.set_fill_color(220, 220, 220)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, 1, 0, "C", 1)
    pdf.ln(8)
    
    pdf.set_fill_color(255, 255, 255)
    fill = False
    for name, value in zip(feature_names, features):
        value_str = str(value) if isinstance(value, str) else f"{float(value):.2f}"
        pdf.cell(col_widths[0], 8, translate(name), 1, 0, "L", fill)
        pdf.cell(col_widths[1], 8, value_str, 1, 0, "L", fill)
        pdf.ln(8)
        fill = not fill
    
    # Recommendations
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, translate("Recommendations"), 0, 1)
    pdf.set_font("Helvetica", size=10)
    next_steps = (
        [translate("Consult a doctor immediately for further tests (e.g., mammogram, biopsy)."),
         translate("Monitor symptoms closely and seek specialist advice.")]
        if result == "Malignant" else
        [translate("Continue regular self-checks and annual screenings."),
         translate("Maintain a healthy lifestyle to reduce risk.")]
    )
    for step in next_steps:
        pdf.cell(0, 8, f"- {step}", 0, 1)
    
    # Footer
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 10, f"PinkScan v1.0 | Generated on {pd.Timestamp.now().strftime('%Y-%m-%d')}", 0, 0, "C")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        pdf.output(temp_file.name)
        return temp_file.name

# Chatbot response function
def get_chatbot_response(user_input):
    user_input = user_input.lower().strip()
    responses = {
        "hello": "Hello! How can I assist you with breast cancer information today?",
        "hi": "Hi there! What would you like to know about breast cancer?",
        "treatment": "Common treatments include surgery, radiation, chemotherapy, and hormone therapy. Consult your doctor for personalized advice.",
        "prevention": "Prevention includes regular screenings, healthy diet, exercise, and avoiding smoking/alcohol.",
        "symptoms": "Symptoms include lumps, changes in breast size/shape, nipple discharge, or skin redness. See a doctor if you notice these.",
        "cure": "Early detection and treatment can manage or eliminate breast cancer. Consult a healthcare professional.",
        "what is breast cancer": "Breast cancer forms in breast cells, often as a lump or tumor. It’s common in women but can affect men too.",
        "help": "I can help with info on symptoms, prevention, treatment, and more. What do you want to know?"
    }
    for keyword in responses:
        if keyword in user_input:
            return translate(responses[keyword])
    return translate("Sorry, I don’t understand. Ask about breast cancer symptoms, prevention, treatment, or say 'help'.")

# Main function
def main():
    logger.info("Starting main function")

    # Language selection
    if 'language' not in st.session_state:
        st.session_state.language = 'English'
    language = st.sidebar.selectbox(translate("Select Language"), options=list(LANGUAGES.values()), 
                                   index=list(LANGUAGES.values()).index(st.session_state.language))
    st.session_state.language = language

    # Title and intro
    st.title(translate("PinkScan - Patient Self-Test"))
    st.write(translate(
        "Assess your breast cancer risk quickly and accurately with this AI-powered tool. Enter your details below for a personalized prediction."
    ))

    # Sidebar - About PinkScan
    st.sidebar.header(translate("About PinkScan"))
    st.sidebar.write(translate(
        "PinkScan uses a Random Forest Classifier trained on key clinical data for rapid, accurate predictions. Malignant: Cancerous tumor. Benign: Non-cancerous tumor."
    ))
    with st.sidebar.expander(translate("Learn More")):
        st.markdown(f"""
            <h3 style='color: #FF69B4;'>{translate("Why PinkScan?")}</h3>
            <p style='color: #333333;'>
                {translate("PinkScan offers fast predictions using 7 key features, enhancing accuracy with personalized inputs like Menopausal Status.")}
            </p>
            <ul style='color: #333333;'>
                <li>{translate("Real-time results")}</li>
                <li>{translate("Easy-to-use interface")}</li>
                <li>{translate("High accuracy with minimal data")}</li>
            </ul>
        """, unsafe_allow_html=True)

    # Patient input
    patient_id = st.text_input(translate("Enter Patient ID"), placeholder="e.g., PAT12345")

    st.header(translate("Your Information"))
    features = []
    with st.expander(translate("Enter Details"), expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input(translate("Age"), min_value=18.0, max_value=100.0, value=30.0, step=1.0)
            family_history = st.selectbox(translate("Family History of Breast Cancer"), options=feature_options["Family History of Breast Cancer"], index=1)
            tumor_size = st.number_input(translate("Tumor Size (mm)"), min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            lymph_nodes = st.selectbox(translate("Lymph Node Involvement"), options=feature_options["Lymph Node Involvement"], index=1)
        with col2:
            menopausal_status = st.selectbox(translate("Menopausal Status"), options=feature_options["Menopausal Status"], index=0)
            breast_pain = st.selectbox(translate("Breast Pain"), options=feature_options["Breast Pain"], index=0)
            skin_changes = st.selectbox(translate("Skin Changes"), options=feature_options["Skin Changes"], index=0)
        features = [age, family_history, tumor_size, lymph_nodes, menopausal_status, breast_pain, skin_changes]

    # Prediction and PDF Download
    if st.button(translate("Predict")):
        if not patient_id:
            st.warning(translate("Please enter a Patient ID."))
        elif model is None or scaler is None:
            st.error(translate("Model or scaler not loaded. Please check the files."))
        else:
            result, risk_score = predict_breast_cancer(features)
            if result != "Error":
                risk_level = "Low" if risk_score < 33.33 else "Medium" if risk_score < 66.67 else "High"
                st.markdown(f"""
                    <div class='prediction-box'>
                        <p>{translate("Patient ID")}: <span>{patient_id}</span></p>
                        <p>{translate("Prediction")}: <span>{translate(result)}</span></p>
                        <p>{translate("Risk Score")}: <span>{risk_score:.2f}% ({translate(risk_level)})</span></p>
                    </div>
                """, unsafe_allow_html=True)

                # Recommendations
                st.subheader(translate("Next Steps"))
                next_steps = (
                    [translate("Consult a doctor immediately for further tests (e.g., mammogram, biopsy)."),
                     translate("Monitor symptoms closely and seek specialist advice.")]
                    if result == "Malignant" else
                    [translate("Continue regular self-checks and annual screenings."),
                     translate("Maintain a healthy lifestyle to reduce risk.")]
                )
                for step in next_steps:
                    st.write(f"- {step}")

                # PDF Download
                st.subheader(translate("Download Your Report"))
                pdf_file = generate_pdf_report(patient_id, result, features, risk_score)
                with open(pdf_file, "rb") as file:
                    st.download_button(
                        label=translate("Download PDF Report"),
                        data=file,
                        file_name=f"report_{patient_id}.pdf",
                        mime="application/pdf",
                        key="download_pdf"
                    )
                os.unlink(pdf_file)  # Clean up temporary file

    # Chatbot Section
    st.header(translate("Ask PinkScan Bot"))
    st.write(translate("Get instant answers about breast cancer from our AI chatbot."))
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with st.container():
        chatbot_container = st.container()
        with chatbot_container:
            for message in st.session_state.chat_history:
                username = translate("You") if message["user"] else translate("PinkScan Bot")
                message_class = "chatbot-user" if message["user"] else "chatbot-bot"
                st.markdown(f'<div class="chatbot-message {message_class}"><strong>{username}:</strong> {message["text"]}</div>', unsafe_allow_html=True)

    user_input = st.text_input(translate("Type your question"), key="chat_input")
    if st.button(translate("Send"), key="chat_send"):
        if user_input:
            st.session_state.chat_history.append({"text": user_input, "user": True})
            response = get_chatbot_response(user_input)
            st.session_state.chat_history.append({"text": response, "user": False})
            st.rerun()

    # Feedback Section
    st.header(translate("Feedback"))
    with st.form("feedback_form"):
        feedback = st.text_area(translate("Provide feedback or suggestions"))
        if st.form_submit_button(translate("Submit Feedback")):
            with open("feedback.txt", "a") as f:
                f.write(f"Patient ID: {patient_id}, Feedback: {feedback}, Date: {pd.Timestamp.now()}\n")
            st.success(translate("Thank you for your feedback!"))

if __name__ == "__main__":
    main()