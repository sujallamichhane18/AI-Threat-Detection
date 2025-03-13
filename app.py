from flask import Flask, request, jsonify
import numpy as np
import joblib

# Initialize Flask App
app = Flask(__name__)

# Load trained models
model = joblib.load('anomaly_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Scale the input features
        features_scaled = scaler.transform(features)

        # Predict using AI model
        prediction = model.predict(features_scaled)
        result = "Anomaly" if prediction[0] == -1 else "Normal"
        
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run API
if __name__ == '__main__':
    app.run(debug=True)
