import streamlit as st
import requests

st.title("Prediksi Average Body Weight (ABW)")

# Gunakan cycle_id otomatis
if "cycle_id" not in st.session_state:
    st.session_state.cycle_id = 1

# Input data
quantity = st.number_input("Quantity (gram)", value=1, min_value=1)
average_weight = st.number_input("Average Weight Shrimp in Sampling", value=1, min_value=1)
weight = st.number_input("Shrimp Harvest Weight (gram)", value=1, min_value=1)
sampled_at = st.text_input("Sampled At (YYYY/MM/DD)", value="2025/03/20")

# ID otomatis berdasarkan jumlah input sebelumnya
cycle_id = st.session_state.cycle_id

if st.button("Predict"):
    api_url = "http://127.0.0.1:5000/predict"
    
    # Data yang akan dikirim ke API
    data = {
        "quantity": quantity,
        "average_weight": average_weight,
        "weight": weight,
        "sampled_at": sampled_at,
        "cycle_id": cycle_id
    }

    # Debugging: Lihat data yang dikirim
    st.write("Data yang dikirim ke API:", data)

    try:
        response = requests.post(api_url, json=data)
        response_json = response.json()

        if response.status_code == 200:
            st.success(f"Prediction: {response_json['prediction']}")
            st.session_state.cycle_id += 1  # Increment cycle_id untuk input berikutnya
        else:
            st.error(f"Error: {response_json}")

    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
