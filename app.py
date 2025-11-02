import streamlit as st
import pandas as pd
import pickle

st.title("Prediksi Kualitas NO₂")

# --- Load model & scaler ---
model = pickle.load(open("model_gnb.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# --- Prediksi manual ---
st.subheader("🔮 Prediksi NO₂ Hari Berikutnya")
col1, col2, col3, col4 = st.columns(4)
t4 = col1.number_input("NO₂ 4 hari sebelumnya", format="%.6f")
t3 = col2.number_input("NO₂ 3 hari sebelumnya", format="%.6f")
t2 = col3.number_input("NO₂ 2 hari sebelumnya", format="%.6f")
t1 = col4.number_input("NO₂ 1 hari sebelumnya", format="%.6f")

if st.button("Prediksi"):
    X_input = pd.DataFrame([[t4, t3, t2, t1]], columns=["t-4", "t-3", "t-2", "t-1"])
    X_input_norm = scaler.transform(X_input)
    prediksi = model.predict(X_input_norm)[0]

    treshold = 0.000543

    if prediksi > treshold:
        st.error(f"Tidak Direkomendasikan")
    else:
        st.success(f"Direkomendasikan")

    st.info(f"Kadar NO₂ yang diprediksi: {prediksi:.6f} µg/m³")
