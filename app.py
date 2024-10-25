import streamlit as st
import numpy as np
import pickle

st.set_page_config("Cardiovascular Classification")

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


st.title(":red[CardioVascular Klasifikasi]")

st.markdown("Aplikasi ini dirancang untuk memprediksi kemungkinan seseorang memiliki risiko penyakit kardiovaskular (penyakit jantung dan pembuluh darah) berdasarkan data kesehatan yang diinput. Dengan menggunakan model machine learning, aplikasi ini dapat memberikan indikasi awal apakah seseorang berisiko mengalami penyakit kardiovaskular, seperti hipertensi atau kolesterol tinggi. Namun, hasil prediksi ini hanya bersifat edukatif, dan untuk hasil diagnosis yang lebih akurat, konsultasikan dengan dokter atau profesional kesehatan.")

# Input data pengguna
age = st.number_input("Umur (dalam hari)", min_value=0, value=20228)
gender = st.selectbox("Jenis Kelamin", options=[0, 1], format_func=lambda x: 'Pria' if x == 1 else 'Wanita')
height = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=70)
ap_hi = st.number_input("Tekanan Sistolik (ap_hi)", min_value=80, max_value=250, value=120)
ap_lo = st.number_input("Tekanan Diastolik (ap_lo)", min_value=40, max_value=150, value=80)
cholesterol = st.selectbox("Kolesterol", options=[1, 2, 3], format_func=lambda x: 'Normal' if x == 1 else 'Tinggi' if x == 2 else 'Sangat Tinggi')
gluc = st.selectbox("Glukosa", options=[1, 2, 3], format_func=lambda x: 'Normal' if x == 1 else 'Tinggi' if x == 2 else 'Sangat Tinggi')
smoke = st.selectbox("Perokok", options=[0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
alco = st.selectbox("Konsumsi Alkohol", options=[0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
active = st.selectbox("Aktif secara fisik", options=[0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')

# Tambahkan tombol untuk memprediksi
if st.button("Prediksi"):
    # Membuat array input berdasarkan data pengguna
    input_data = np.array(scale.transform([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]]))
    
    # Prediksi menggunakan model
    prediction = model.predict(input_data)

    # Outputkan hasil prediksi
    if prediction[0] == 1:
        st.error("Hasil Prediksi: Anda kemungkinan memiliki risiko penyakit kardiovaskular.")
    else:
        st.success("Hasil Prediksi: Anda tidak memiliki risiko penyakit kardiovaskular.")
