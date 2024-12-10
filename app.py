import streamlit as st
import numpy as np
import pickle
import datetime as datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="CardioVascular Risk Prediction", layout="wide")

def load_model(): 
    with open("FINAL_MODEL_CARDIO.pkl", 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def load_scale(): 
    with open("SCALED.pkl", 'rb') as file:
        scale = pickle.load(file)
    return scale

model = load_model()
scale = load_scale()

st.title(":red[CardioVascular Risk Classification]")

st.markdown(""" 
    **Welcome to the CardioVascular Risk Prediction Tool!**  
    This tool predicts the likelihood of cardiovascular diseases based on your health data.  
""", unsafe_allow_html=True)

st.markdown("---")
st.header("Enter your details")

with st.form(key="input_form"):
    birth_date = st.date_input("Birth Date", max_value=datetime.date.today())
    today = datetime.date.today()
    age_in_days = (today - birth_date).days
    age_in_years = age_in_days // 365
    st.write(f"Age: {age_in_years} years")
    st.write(f"Age: {age_in_days} days")
    
    age = age_in_days
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
    height = st.number_input("Height (in cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (in kg)", min_value=30, max_value=200, value=70)
    ap_hi = st.number_input("Systolic Pressure (ap_hi)", min_value=80, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic Pressure (ap_lo)", min_value=40, max_value=150, value=80)
    cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3], format_func=lambda x: 'Normal' if x == 1 else 'High' if x == 2 else 'Very High')
    gluc = st.selectbox("Glucose Level", options=[1, 2, 3], format_func=lambda x: 'Normal' if x == 1 else 'High' if x == 2 else 'Very High')
    smoke = st.selectbox("Do you smoke?", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    alco = st.selectbox("Do you consume alcohol?", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    active = st.selectbox("Do you exercise regularly?", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

    submit_button = st.form_submit_button(label="Predict Risk")

if submit_button:
    input_data = np.array(scale.transform([[age_in_days, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]]))
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("**Prediction Result**: You are at a higher risk of cardiovascular disease.")
        st.markdown("### What this means:")
        st.write("A higher risk of cardiovascular disease could mean the presence of conditions like hypertension or high cholesterol.")
    else:
        st.success("**Prediction Result**: You are at a lower risk of cardiovascular disease.")
        st.markdown("### What this means:")
        st.write("A lower risk suggests that your health factors are within a generally healthy range.")

    bmi = weight / (height / 100) ** 2
    st.write(f"Your BMI: {bmi:.2f}")
    if bmi < 18.5:
        st.warning("You are underweight.")
    elif 18.5 <= bmi < 24.9:
        st.success("Your weight is in the normal range.")
    elif 25 <= bmi < 29.9:
        st.warning("You are overweight.")
    else:
        st.error("You are obese.")

    st.markdown("### Health Tips:")
    if cholesterol == 3:
        st.write("- Reduce intake of high-cholesterol foods.")
    if ap_hi > 130:
        st.write("- Manage your blood pressure with diet and exercise.")
    if smoke == 1:
        st.write("- Quitting smoking can reduce your cardiovascular risk.")
    if not active:
        st.write("- Regular exercise can improve cardiovascular health.")

    st.markdown("### Blood Pressure Visualization")
    fig, ax = plt.subplots(figsize=(8, 5))  # Smaller figure size
    ax.bar(["Systolic (ap_hi)", "Diastolic (ap_lo)"], [ap_hi, ap_lo], color=["red", "blue"])
    ax.set_ylabel("Pressure (mmHg)")
    ax.set_title("Blood Pressure")
    st.pyplot(fig)

st.markdown("""
    <footer style="text-align:center; margin-top:50px;">
        <p>Created by Hans Santoso</p>
        <p><a href="https://www.linkedin.com/in/hans-santoso/" target="_blank">Connect with me on LinkedIn</a></p>
    </footer>
""", unsafe_allow_html=True)   
