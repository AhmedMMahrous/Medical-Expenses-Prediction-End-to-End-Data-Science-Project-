
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the model from the saved file
scaler = joblib.load('scaler.save')
model = joblib.load('Medical Expenses Prediction.save')

# home
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


# predict
@app.route('/predict', methods=['GET','POST'])
def predict():
     if request.method == 'POST':  # while prediction
        age = request.form['age']
        sex = request.form['sex']
        bmi = request.form['bmi']
        children = request.form['children']
        smoker = request.form['smoker']
        region = request.form['region']
        inp_data = [age,sex,bmi,children,smoker,region]
        inp_data = [(int(x)) for x in inp_data]
        prediction = model.predict(scaler.transform([inp_data]))
        prediction = '{:.4f}'.format(prediction[0])

        return render_template('predict.html' ,pred_val = prediction)
     else:
        return render_template('predict.html')
   
 
# about
@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

# The function called when the script is run
if __name__ == '__main__':
    app.run(debug=True)



    



