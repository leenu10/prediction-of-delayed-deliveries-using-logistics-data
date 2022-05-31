# Importing the required libraries and modules.
import pickle
import numpy as np
from flask import Flask, render_template, request

# reading the model from the pickle file
model = pickle.load(open('model.pickle', 'rb'))

# Checking a prediction using the model.
prediction = model.predict([np.array([1,7,5,34,2,0,4,6,0,90,0,0,0,26.326825])])
print(prediction)

# Initiating Flask app.
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features_final = [np.array(features)]
    prediction = model.predict(features_final)
    if prediction == 0:
        prediction_text = "Oops! Your Delivery will be Delayed..☹️"
    else:
        prediction_text = 'Your Delivery will be On time.😀'
       
    return render_template('result.html', prediction_text = prediction_text.format(prediction))
if __name__ == '__main__':
    app.run()