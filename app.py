from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the trained model
with open('Rf_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open('Scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the label encoder
with open('Label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    weight = float(request.form['weight'])
    length1 = float(request.form['length1'])
    length2 = float(request.form['length2'])
    length3 = float(request.form['length3'])
    height = float(request.form['height'])
    width = float(request.form['width'])
    
    # Create a feature array
    features = np.array([[weight, length1, length2, length3, height, width]])
    
    # Standardize the features
    features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features)
    species = label_encoder.inverse_transform(prediction)
    
    return f'This could be a {species[0]}'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)