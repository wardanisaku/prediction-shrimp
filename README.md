# Data Evaluation, Predictive Model, API, and Streamlit Interface

## About the Project

This project provides a structured approach to:
- Evaluate data completeness.
- Develop a predictive model using XGBoost.
- Deploy APIs for model inference using Flask.
- Build a Streamlit interface for user-friendly interaction.

## Features

- **Data Evaluation**: Ensures data consistency and completeness.
- **Predictive Modeling**: Uses XGBoost for survival rate and Average Body Weight predictions.
- **Flask API**: Serves predictions via REST endpoints.
- **Streamlit UI**: Provides an intuitive web interface for input and results.

## Data Evaluation

To ensure high-quality input data, the following steps are implemented:

1. **Check for Missing Values**:
   - Identify missing data using `df.isnull().sum()`.
   - Handle missing values via imputation or removal.

2. **Verify Data Types and Consistency**:
   - Convert date fields to `datetime` format.
   - Ensure numerical fields are correctly formatted.

3. **Detect Outliers**:
   - Analyze distributions using `df.describe()`.
   - Use boxplots to visualize anomalies.

4. **Check Unique Values**:
   - Identify categorical columns and validate unique values.

## Predictive Model

The predictive model uses a trained XGBoost algorithm to estimate:

- **Survival Rate**
- **Average Body Weight (ABW)**

### Preprocessing Steps:
1. Feature Engineering:
   - Compute `survival_rate`, `adg`, and `fcr`.
2. Scaling:
   - Normalize data using a pre-trained scaler.
3. Model Prediction:
   - Load the `xgboost_model.pkl` and apply `.predict()`.

## API Implementation

Flask serves two prediction APIs:

- **API 1: Survival Rate Prediction**
- **API 2: ABW Prediction**

### API Endpoints:
- `GET /` → Health check
- `POST /predict` → Returns model predictions

### API Preprocessing:
- API 1 Features: `weight`, `total_seed`, `adg`, `selling_price`, `morning_do`.
- API 2 Features: `fcr`, `quantity`, `adg`.

## Streamlit Interface

The Streamlit UI provides an intuitive web-based interaction:

- **Survival Rate Prediction Interface**
- **ABW Prediction Interface**

### UI Features:
- Uses `st.number_input()` for numeric inputs.
- Displays real-time API responses.
- Handles connection errors gracefully.

## Contributing

Contributions are welcome! Please check the [contribution guidelines](https://github.com/your-repo/contribute).

## License

This project is open-source under the [MIT License](https://opensource.org/licenses/MIT).

