import json
import numpy as np
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Loading models...")
    model = joblib.load('anomaly_model.pkl')
    scaler = joblib.load('scaler.pkl')
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise

def handler(event, context):
    logger.info("Received request.")
    try:
        body = json.loads(event.get('body', '{}'))
        features = np.array(body.get('features', [])).reshape(1, -1)

        logger.info(f"Feature length: {len(features[0])}")
        if len(features[0]) != 78:
            logger.warning(f"Expected 78 features, got {len(features[0])}")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Expected 78 features, got ' + str(len(features[0]))})
            }

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        result = "Anomaly" if prediction[0] == -1 else "Normal"
        logger.info(f"Prediction: {result}")

        return {
            'statusCode': 200,
            'body': json.dumps({'prediction': result})
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }s
