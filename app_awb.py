import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model dan scaler
model = joblib.load("xgboost_model_awb.pkl")
scaler = joblib.load("scaler_awb.pkl")

features = ['fcr', 'quantity', 'adg']

class Preprocessor:
    def __init__(self, data):
        self.data = data.copy()

    def calculate_adg(self):
        if {'cycle_id', 'sampled_at', 'weight'}.issubset(self.data.columns):
            self.data['sampled_at'] = pd.to_datetime(self.data['sampled_at'], errors="coerce")
            self.data = self.data.dropna(subset=['sampled_at'])
            self.data = self.data.sort_values(['cycle_id', 'sampled_at'])

            self.data['adg'] = self.data.groupby('cycle_id')['weight'].diff() / \
                               self.data.groupby('cycle_id')['sampled_at'].diff().dt.days

            self.data['adg'] = self.data['adg'].fillna(0)
        return self
    
    def calculate_fcr(self):
        if {'quantity', 'average_weight'}.issubset(self.data.columns):
            self.data['fcr'] = np.where(
                self.data['average_weight'] > 0,
                self.data['quantity'] / self.data['average_weight'],
                np.nan
            )
            self.data['fcr'] = self.data['fcr'].fillna(0)  
        return self

    def select_features(self):
        self.data = self.data[features]
        return self

    def get_processed_data(self):
        return self.data

@app.route("/")
def home():
    return "API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if isinstance(data, dict):
            data = [data]

        input_df = pd.DataFrame(data)

        preprocessor = Preprocessor(input_df)
        processed_data = (
            preprocessor
            .calculate_fcr() 
            .calculate_adg()
            .select_features()
            .get_processed_data()
        )

        if processed_data.isnull().values.any():
            return jsonify({"error": "Data contains NaN values after preprocessing"}), 400

        input_data_scaled = scaler.transform(processed_data)

        prediction = model.predict(input_data_scaled).tolist()

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
