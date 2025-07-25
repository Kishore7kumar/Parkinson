import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Parkinson’s Detection from Numeric Voice Features")

feature_str = st.text_area("Paste the numeric feature vector (comma separated):")

if st.button("Predict"):
    try:
        features = np.array([float(x.strip()) for x in feature_str.split(",")])
        
        if features.size != 28:
            st.error(f"Expected 28 features, got {features.size}. Please check your input.")
        else:
            features = features.reshape(1, -1)
            scaled = scaler.transform(features)
            pred = model.predict(scaled)[0]
            result = "⚠️ Parkinson’s Detected" if pred == 1 else "🟩 No Parkinson’s Detected"
            st.success(f"Prediction: {result}")
            
    except Exception as e:
        st.error(f"Invalid input or prediction error: {e}")

