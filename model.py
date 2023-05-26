from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('front.html')


@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    sex = int(request.form.get('sex'))
    cp = int(request.form.get("cp"))
    trestbps = int(request.form.get("trestbps"))
    chol = int(request.form.get("chol"))
    fbs = int(request.form.get("fbs"))
    restecg = int(request.form.get("restecg"))
    thalach = int(request.form.get("thalach"))
    exang = int(request.form.get("exang"))
    oldpeak = int(request.form.get("oldpeak"))
    slope = int(request.form.get("slope"))
    ca = int(request.form.get("ca"))
    thal = int(request.form.get("thal"))
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    print(prediction)
    if prediction == 1:
        return render_template('front.html', label=1)
    else:
        return render_template('front.html', label=-1)


if __name__ == '__main__':
    app.run(debug=True)
