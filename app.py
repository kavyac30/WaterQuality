from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and label encoder
model = joblib.load('water_quality_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('home.html')  # Landing page

@app.route('/predict_form')
def predict_form():
    return render_template('index.html')  # Prediction form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['pH']),
            float(request.form['Dissolved_Oxygen']),
            float(request.form['Salinity']),
            float(request.form['Secchi_Depth']),
            float(request.form['Water_Depth']),
            float(request.form['Water_Temp']),
            float(request.form['Air_Temp'])
        ]

        input_df = pd.DataFrame([features], columns=[
            'pH',
            'Dissolved_Oxygen',
            'Salinity',
            'Secchi_Depth',
            'Water_Depth',
            'Water_Temp',
            'Air_Temp'
        ])
        
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        return render_template('index.html', prediction_text=f"Predicted Water Quality: {predicted_label}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
