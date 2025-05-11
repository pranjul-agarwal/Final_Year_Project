from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load ML models and preprocessing tools
fog_model = joblib.load('fog_classification_model.pkl')
fog_scaler = joblib.load('scaler.pkl')
fog_label_encoder = joblib.load('label_encoder.pkl')

# Load regression models and scalers
temperature_model = joblib.load('TEMPERATURE_model.pkl')
temperature_scaler = joblib.load('temp_scaler.pkl')
X_train_scaled_temp = joblib.load('TEMP_X_train_scaled.pkl')  # For ANOVA kernel prediction

pm25_model = joblib.load('pm25_model.pkl')
pm25_scaler = joblib.load('pm25_scaler.pkl')
X_train_scaled_25 = joblib.load('PM25_X_train_scaled.pkl')  # For ANOVA kernel prediction

pm10_model = joblib.load('pm10_model.pkl')
pm10_scaler = joblib.load('pm10_scaler.pkl')

aqi_model = joblib.load('AQI_model.pkl')
aqi_scaler = joblib.load('AQI_scaler.pkl')

humidity_model = joblib.load('rh_model.pkl')
humidity_scaler = joblib.load('rh_scaler.pkl')
X_train_scaled_RH = joblib.load('RH_X_train_scaled.pkl')  # For ANOVA kernel prediction

# ANOVA kernel function
def anova_kernel(X1, X2, gamma=0.1, d=2):
    kernel_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[1]):
        feature_diff = np.subtract.outer(X1[:, i], X2[:, i])
        kernel_matrix += np.exp(-gamma * (feature_diff ** 2)) ** d
    return kernel_matrix

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        temp = float(data['temp'])
        pm25 = float(data['pm25'])
        aqi = float(data['aqi'])
        pm10 = float(data['pm10'])
        humidity = float(data['humidity'])

        # Input in model's expected order
        model_input = np.array([[aqi, pm10, pm25, temp, humidity]])
        scaled_input = fog_scaler.transform(model_input)
        prediction = fog_model.predict(scaled_input)
        result = fog_label_encoder.inverse_transform(prediction)[0]

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_regression', methods=['POST'])
def predict_regression():
    data = request.form
    parameter = data['parameter']
    file = request.files['file']

    try:
        # Read the uploaded Excel file 
        remove_columns = ['Date','FOG_CATEGORY']
        df = pd.read_excel(file, sheet_name="Sheet1")
        # Remove unwanted columns if they exist
        df = df.drop(columns=[col for col in remove_columns if col in df.columns])

        # Remove the selected column if present
        if parameter in df.columns:
            df = df.drop(columns=[parameter])

        # Get the right model, scaler, and (optionally) X_train_scaled
        if parameter == 'Temperature(°C)':
            model = temperature_model
            scaler = temperature_scaler
            X_train_scaled = X_train_scaled_temp
            is_anova = True
        elif parameter == 'PM 2.5(µg/m³)':
            model = pm25_model
            scaler = pm25_scaler
            X_train_scaled = X_train_scaled_25
            is_anova = True
        elif parameter == 'PM 10(µg/m³)':
            model = pm10_model
            scaler = pm10_scaler
            is_anova = False
        elif parameter == 'AQI':
            model = aqi_model
            scaler = aqi_scaler
            is_anova = False
        elif parameter == 'Relative_humidity(%)':
            model = humidity_model
            scaler = humidity_scaler
            X_train_scaled = X_train_scaled_RH
            is_anova = True
        else:
            return jsonify({'error': 'Invalid parameter selected'})

        # Scale the input data
        scaled_data = scaler.transform(df)

        # Predict with or without ANOVA kernel
        if is_anova:
            # Compute the ANOVA kernel between new data and training data
            K_val = anova_kernel(scaled_data, X_train_scaled)
            predictions = model.predict(K_val)
        else:
            predictions = model.predict(scaled_data)

        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
