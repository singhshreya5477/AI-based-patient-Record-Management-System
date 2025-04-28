# app.py

import streamlit as st
import pandas as pd
import joblib
import csv
from datetime import datetime

# --- Load Models and Data ---
model = joblib.load('models/symptom_disease_model.pkl')
le = joblib.load('models/label_encoder.pkl')
all_symptoms = joblib.load('models/symptom_list.pkl')

# Load additional CSVs
description_df = pd.read_csv('data/disease_symptom_description.csv')
description_df = description_df.iloc[:, :2]
description_df.columns = ['Disease', 'Description']

precaution_df = pd.read_csv('data/symptom_precaution.csv')
precaution_df.columns = precaution_df.columns.str.strip()
precaution_df.columns = ['Disease', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']

test_recommendation_df = pd.read_csv('data/disease_test_mapping.csv')
test_recommendation_df.columns = test_recommendation_df.columns.str.strip()
test_recommendation_df.columns = ['Disease', 'Test_1', 'Test_2', 'Test_3']

# --- Define Functions ---
def predict_disease(user_symptoms):
    input_vector = [1 if symptom in user_symptoms else 0 for symptom in all_symptoms]
    encoded_prediction = model.predict([input_vector])[0]
    disease = le.inverse_transform([encoded_prediction])[0]
    return disease

def get_description(disease):
    row = description_df[description_df['Disease'].str.lower() == disease.lower()]
    return row['Description'].values[0] if not row.empty else "No description available."

def get_precautions(disease):
    row = precaution_df[precaution_df['Disease'].str.lower() == disease.lower()]
    return row.iloc[0, 1:].dropna().tolist() if not row.empty else ["No precautions available."]

def get_test_recommendations(disease):
    row = test_recommendation_df[test_recommendation_df['Disease'].str.lower() == disease.lower()]
    return row.iloc[0, 1:].dropna().tolist() if not row.empty else ["No specific test recommendation available."]

def save_patient_record(name, age, gender, symptoms, predicted_disease):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = [now, name, age, gender, ', '.join(symptoms), predicted_disease]
    with open('data/patient_records.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(record)

# --- Streamlit UI Settings ---
st.set_page_config(page_title="AI-Based Patient Record Management System", page_icon="🩺", layout="wide")

# Sidebar
st.sidebar.title("🩺 Patient Information")

name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Patient Age", min_value=0, max_value=120)
gender = st.sidebar.selectbox("Patient Gender", ["Male", "Female", "Other"])

# Main Title
st.title("AI-Based Disease Diagnosis System")
st.markdown("Fill patient info in sidebar ➡️ Select symptoms ➡️ Predict disease ➡️ View recommendations!")

# Symptoms Selection
st.subheader("🤒 Select Symptoms")
selected_symptoms = st.multiselect(
    "Select Symptoms",
    all_symptoms,
    help="Start typing to search symptoms..."
)

# --- Predict Disease Button ---
if st.button("🔍 Predict Disease"):
    if not name or not selected_symptoms:
        st.warning("⚠️ Please enter patient's name and select at least one symptom.")
    else:
        disease = predict_disease(selected_symptoms)
        st.success(f"🧠 Predicted Disease: **{disease}**")

        st.subheader("📘 Disease Description")
        st.write(get_description(disease))

        st.subheader("🛡️ Precautions")
        for precaution in get_precautions(disease):
            st.write(f"✅ {precaution}")

        st.subheader("🧪 Recommended Tests")
        for test in get_test_recommendations(disease):
            st.write(f"🔬 {test}")

        # Save the record
        save_patient_record(name, age, gender, selected_symptoms, disease)
        st.success("✅ Patient record saved successfully!")

# --- View Patient Records ---
st.markdown("---")
st.subheader("📋 View Saved Patient Records")

if st.button("📂 Show All Patient Records"):
    try:
        patient_df = pd.read_csv('data/patient_records.csv')
        st.dataframe(patient_df)
    except FileNotFoundError:
        st.error("❌ No patient records found yet. Make some predictions first!")

# --- Footer ---
st.markdown("---")
st.caption("Developed by [Your Name] | AI-Based Patient Record Management System © 2025")
