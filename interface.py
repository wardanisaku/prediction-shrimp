import streamlit as st
import requests

st.title("Predictions Survival Rate %")

if "cycle_id" not in st.session_state:
    st.session_state.cycle_id = 1

weight = st.number_input("Shrimp Harvest Weight (gram)", value=1)
total_seed = st.number_input("Total Seed", value=1)
selling_price = st.number_input("Selling Price (IDR)", value=1)
morning_do = st.number_input("Morning DO", value=1)
size = st.number_input("Size Shirmp (tail/kg)", value=1)
sampled_at = st.text_input("Sampled At", value="2025/03/20")

# Gunakan cycle_id dari session state
cycle_id = st.session_state.cycle_id

if st.button("Predict"):
    api_url = "http://127.0.0.1:5000/predict"
    data = {
        "weight": weight,
        "total_seed": total_seed,
        "size": size,
        "selling_price": selling_price,
        "morning_do": morning_do,
        "cycle_id": cycle_id,  # Gunakan ID otomatis
        "sampled_at": sampled_at
    }

    # Debugging: Lihat data yang dikirim
    st.write("Data yang dikirim ke API:", data)

    try:
        response = requests.post(api_url, json=data)
        response_json = response.json()

        if response.status_code == 200:
            st.success(f"Prediction: {response_json['prediction']}")
            st.session_state.cycle_id += 1  # Tambah cycle_id untuk input berikutnya
        else:
            st.error(f"Error: {response_json}")

    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
