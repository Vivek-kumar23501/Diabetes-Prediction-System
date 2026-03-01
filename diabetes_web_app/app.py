from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# model load
classifier = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = (
        float(request.form['preg']),
        float(request.form['glu']),
        float(request.form['bp']),
        float(request.form['skin']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['dpf']),
        float(request.form['age'])
    )

    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    input_data_reshaped = pd.DataFrame(input_data_reshaped, columns=scaler.feature_names_in_)

    std_data=scaler.transform(input_data_reshaped)
    prediction=classifier.predict(std_data)

    if prediction[0]==0:
        result = "The person is not diabetic"
    else:
        result = "The person is diabetic"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)