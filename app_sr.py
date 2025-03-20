import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model dan scaler
model = joblib.load("xgboost_model_sr.pkl")
scaler = joblib.load("scaler_sr.pkl")

features = ['weight', 'total_seed', 'adg', 'selling_price', 'morning_do']

class Preprocessor:
    def __init__(self, data):
        self.data = data.copy()

    def calculate_survival_rate(self):
        if {'total_seed', 'size', 'weight'}.issubset(self.data.columns):
            self.data['survival_rate'] = np.where(
                self.data['total_seed'] > 0,  
                (self.data['size'] * self.data['weight'] / self.data['total_seed']) * 100,
                np.nan
            )
            self.data['survival_rate'] = self.data['survival_rate'].round(2)
        return self

    def calculate_adg(self):
        if {'cycle_id', 'sampled_at', 'weight'}.issubset(self.data.columns):
            self.data['sampled_at'] = pd.to_datetime(self.data['sampled_at'], errors="coerce")
            self.data = self.data.dropna(subset=['sampled_at'])
            self.data = self.data.sort_values(['cycle_id', 'sampled_at'])

            self.data['adg'] = self.data.groupby('cycle_id')['weight'].diff() / \
                               self.data.groupby('cycle_id')['sampled_at'].diff().dt.days

            self.data['adg'] = self.data['adg'].fillna(0)
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
            .calculate_survival_rate()
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
