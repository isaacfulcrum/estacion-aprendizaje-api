from flask import Flask, request
import pickle

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    # Make a prediction using the model
    prediction = model.predict([list(data.values())])
    # Return the prediction
    return str(prediction[0])


if __name__ == '__main__':
    app.run(port=5000, debug=True)