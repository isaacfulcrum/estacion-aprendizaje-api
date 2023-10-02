from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import openai
from performance_trend import get_trend
from util import parse_scores

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


app = Flask(__name__)
# Load the model
model = joblib.load('bin/model.joblib')
label_encoder = joblib.load('bin/label_encoder.joblib')  # Assuming you saved label_encoder as well


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Preprocess the data
    scores = parse_scores(data.get('prev_scores', ''))  # Assuming 'prev_scores' is in the incoming JSON data
    processed_data = {
        'prev_scores_mean': np.mean(scores) if scores else 0.0,
        'prev_scores_max': np.max(scores) if scores else 0.0,
        'prev_scores_min': np.min(scores) if scores else 0.0,
        'prev_scores_std': np.std(scores) if scores else 0.0,
        'score': data.get('score', 0.0)  # Assuming 'score' is in the incoming JSON data
    }
    # Make prediction
    prediction = model.predict(np.array(list(processed_data.values())).reshape(1, -1))
    # Convert prediction back to the original class
    predicted_class = label_encoder.inverse_transform(prediction)

    updated_scores = scores + [processed_data['score']]
    trend = get_trend(updated_scores).name.replace('_', ' ').title()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": f'Given that the user\'s performance trend is "{trend}", craft an encouraging message to motivate them to continue their progress. Be concise and use natural language. Don\'t overdue it with the praise. '
                        f'The message should be no longer than 100 characters. And it should be in SPANISH. '},
        ]
    )

    generated_message = response['choices'][0]['message']['content']

    return jsonify({'level': predicted_class.tolist()[0], 'message': generated_message })


if __name__ == '__main__':
    app.run(debug=True)