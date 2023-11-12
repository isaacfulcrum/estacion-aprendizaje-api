from flask import Flask, request, jsonify
import joblib
import numpy as np
from util import parse_scores
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model
model = joblib.load('bin/model.joblib')
label_encoder = joblib.load('bin/label_encoder.joblib')  # Assuming you saved label_encoder as well


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Preprocess the data
    scores = parse_scores(data.get('scores', ''))  # Assuming 'scores' is in the incoming JSON data
    processed_data = {
        'scores_mean': np.mean(scores) if scores else 0.0,
        'scores_max': np.max(scores) if scores else 0.0,
        'scores_min': np.min(scores) if scores else 0.0,
        'scores_std': np.std(scores) if scores else 0.0,
    }
    # Make prediction
    prediction = model.predict(np.array(list(processed_data.values())).reshape(1, -1))
    # Convert prediction back to the original class
    predicted_class = label_encoder.inverse_transform(prediction)

    return jsonify({'level': predicted_class.tolist()[0]})


if __name__ == '__main__':
    app.run(debug=True, port=3001)