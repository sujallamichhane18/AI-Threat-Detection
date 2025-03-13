import json
import numpy as np
import joblib

# Load trained models
model = joblib.load('anomaly_model.pkl')
scaler = joblib.load('scaler.pkl')

def handler(event, context):
    try:
        # Get input data from the event body
        body = json.loads(event.get('body', '{}'))
        features = np.array(body.get('features', [])).reshape(1, -1)

        # Validate feature length
        if len(features[0]) != 78:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Expected 78 features, got ' + str(len(features[0]))})
            }

        # Scale the input features
        features_scaled = scaler.transform(features)

        # Predict using AI model
        prediction = model.predict(features_scaled)
        result = "Anomaly" if prediction[0] == -1 else "Normal"

        return {
            'statusCode': 200,
            'body': json.dumps({'prediction': result})
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
