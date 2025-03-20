from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan scaler
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# Fitur yang digunakan
features = ['weight', 'total_seed', 'pond_id_measurements', 'morning_do', 'z_score']

@app.route("/")
def home():
    return "API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Pastikan semua fitur ada
        if not all(f in data for f in features):
            return jsonify({"error": "Missing features"}), 400

        # Ambil data dari request
        input_data = np.array([[data[f] for f in features]])

        # Scale data
        input_data_scaled = scaler.transform(input_data)

        # Prediksi
        prediction = model.predict(input_data_scaled)[0]

        return jsonify({"prediction": float(prediction)})  # Konversi float32 ke float

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
