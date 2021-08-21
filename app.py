#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(prediction)
    output = int(prediction[0])
    print(output, type(output))

    if output == 1:
        prediction_pos = "The patient is diagnosed with high possibilities of having a heart disease. Consult a Cardiologist!"
        return render_template('index.html', prediction_pos = prediction_pos)
    elif output == 0:
        prediction_neg = "The patient is not diagnosed with any cardiac problem. Patient is safe!"
        return render_template('index.html', prediction_neg = prediction_neg)


    

if __name__ == "__main__":
    app.run(debug=True)