import pickle
from flask import Flask,render_template,request
import numpy as np
model_path='Diabetes Prediction Model\model.pkl'
scalar_path='Diabetes Prediction Model\scaler.pkl'

with open(model_path,'rb') as model_file:
    model=pickle.load(model_file)
with open(scalar_path,'rb') as scalar_file:
    scalar=pickle.load(scalar_file)

app=Flask(__name__)
@app.route('/')
def about():
    return render_template('about.html')
@app.route('/model')
def home():
   return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    values=list(request.form.values())
    Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age = values
    Pregnancies = int(Pregnancies)
    Glucose = int(Glucose)
    BloodPressure = int(BloodPressure)
    SkinThickness = int(SkinThickness)
    Insulin = int(Insulin)
    BMI = float(BMI)
    DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
    Age = int(Age)
    data=np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]).reshape(1,-1)
    data=scalar.transform(data)
    prediction=model.predict(data)
    msg="Person is Diabetic" if prediction[0]==1 else "Person is not Diabetic"
    return render_template('prediction.html',text=msg)

if __name__=='__main__':
    app.run(debug=True)
