import numpy as np
from flask import Flask, request, jsonify, render_template
from model import predict_speech
import pickle
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

app = Flask(__name__)
model = pickle.load(open('model_dt_clf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    to_predict_list = request.form.to_dict()

    to_predict_list = list(to_predict_list.values())
    prediction = predict_speech(str(to_predict_list))




    return render_template('index.html', prediction_text='Predicted Speech Type {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
