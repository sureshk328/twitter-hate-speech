import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

app = Flask(__name__)
model_logreg = pickle.load(open('model_logreg.pkl', 'rb'))
model_knn = pickle.load(open('model_knn.pkl', 'rb'))
model_svm = pickle.load(open('model_svm.pkl', 'rb'))
model_dectree = pickle.load(open('model_dectree.pkl', 'rb'))
cv = pickle.load(open('transform.pkl','rb'))

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
    data = cv.transform([str(to_predict_list)]).toarray()
    prediction_logreg = "Logistic Regression: "+model_logreg.predict(data)
    prediction_knn = "KNN: "+model_knn.predict(data)
    prediction_svm = "Support Vector Machine: "+model_svm.predict(data)
    prediction_dectree = "DecisionTreeClassifier: "+ model_dectree.predict(data)






    #return render_template('index.html', prediction_text='Predicted Speech Type {}'.format(prediction))
    return render_template('index.html',prediction_text="Below are prediction results: ", prediction_text_logreg=prediction_logreg, prediction_text_knn=prediction_knn,prediction_text_svn=prediction_svm, prediction_text_dectree=prediction_dectree)


if __name__ == "__main__":
    app.run(debug=True)
