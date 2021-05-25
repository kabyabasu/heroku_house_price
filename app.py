from flask import Flask,request,render_template,jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():

    list_input = [x for x in request.form.values()]
    _features = [np.array(list_input)]
    print(_features)
    _output = model.predict(_features)

    return  render_template('index.html', prediction_text = "House price is {}".format(_output))


if __name__ == "__main__":
    app.run(debug=True)

