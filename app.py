import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import joblib

# Load the trained model
symptoms_model = joblib.load('trained_model.pickle')
# Load your image-based model
model_resnet = load_model("resnet50_model.h5")

# Define categories for image-based prediction
categories=['cervix_dyk','cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi']


def predict_with_image():
    # Let the user upload an image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file:
        img = image.load_img(uploaded_file, target_size=(64, 64))

        # Convert the image to a numpy array and add a batch dimension
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Preprocess the image for the model
        x = preprocess_input(x)

        # Make a prediction
        predictions = model_resnet.predict(x)

        # Decode probabilities into class labels
        class_label = categories[np.argmax(predictions)]
        st.write(f'Predicted class: {class_label}')

def predict_with_symptoms():
    age = st.slider('Age', 0, 100)
    gender = st.selectbox('Gender', [1, 2]) # Assuming 1 for male and 2 for female
    blood_pressure = st.number_input('Blood Pressure', value=120, min_value=0, max_value=250)
    glucose_level = st.number_input('Glucose Level', value=100, min_value=0, max_value=500)
    fatigue = st.slider('Fatigue', 0, 10)
    smoking = st.selectbox('Smoking', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sores = st.slider('Sores', 0, 10)
    thick_lump = st.number_input('Thick Lump', value=0, min_value=0, max_value=500)
    insulin_level = st.number_input('Insulin Level', value=0, min_value=0, max_value=500)
    skin_thickness = st.number_input('Skin Thickness', value=0, min_value=0, max_value=100)
    chest_pain = st.slider('Chest Pain', 0, 10)
    weight_loss = st.slider('Weight Loss', 0, 10)
    balanced_diet = st.slider('Balanced Diet', 0, 10)
    occupational_hazards = st.slider('Occupational Hazards', 0, 10)
    dust_allergy = st.slider('Dust Allergy', 0, 10)
    shortness_of_breath = st.slider('Shortness of Breath', 0, 10)
    swallowing_difficulty = st.slider('Swallowing Difficulty', 0, 10)
    frequent_cold = st.slider('Frequent Cold', 0, 10)


    # Aggregate inputs
    user_data = [age, gender, blood_pressure, glucose_level, fatigue, smoking, sores, thick_lump, insulin_level, skin_thickness, chest_pain, weight_loss, balanced_diet, occupational_hazards, dust_allergy, shortness_of_breath, swallowing_difficulty, frequent_cold]

    # Predict using the symptoms model
    prediction_proba = symptoms_model.predict_proba(np.array([user_data]))[0]
    
    # Here I'm assuming class "1" is for "Cancer". Adjust as needed.
    cancer_confidence = prediction_proba[1] * 100

    # Display the confidence score
    st.write(f'The model is {cancer_confidence:.2f}% confident that the symptoms indicate cancer.')

# App layout
st.title('Prediction App')
choice = st.selectbox("Choose Prediction Type", ["Predict with Image", "Predict with Symptoms"])

if choice == "Predict with Image":
    predict_with_image()
elif choice == "Predict with Symptoms":
    predict_with_symptoms()