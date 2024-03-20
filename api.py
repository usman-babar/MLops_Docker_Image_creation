from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load("Diabetes.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Make prediction
        prediction = model.predict([data])[0]
        
        # Return prediction
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
